#include "storage/glue_hive_insert.hpp"

#include "duckdb/catalog/catalog_entry/copy_function_catalog_entry.hpp"
#include "duckdb/common/hive_partitioning.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/string_functions.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"

#include "glue_api.hpp"
#include "storage/glue_catalog.hpp"
#include "storage/glue_schema_entry.hpp"
#include "storage/glue_table.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Planning
//===--------------------------------------------------------------------===//
GlueHiveInsert::GlueHiveInsert(PhysicalPlan &physical_plan, LogicalOperator &op, GlueTable &table, bool discard,
                               GlueHiveWriteInfo write_info)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, {LogicalType::BIGINT}, op.estimated_cardinality),
      table(table), discard(discard), write_info(std::move(write_info)) {
}

namespace {

//! Where one registered partition is written, or why it cannot be
struct GluePartitionDirectory {
	//! The partition's directory relative to the table location
	string directory;
	//! Set when DuckDB's partitioned writer cannot express this partition's location.
	string refusal;
};

//! The registered directory of each partition, keyed by its values.
struct GluePartitionDirectories : public ScalarFunctionInfo {
	//! Partition values (VARCHAR, NUL-separated) to where that partition is written
	unordered_map<string, GluePartitionDirectory> by_values;
};

//! Glue stores partition values as strings, so the key is the values rendered as VARCHAR.
string PartitionValuesKey(const vector<string> &values) {
	string key;
	for (auto &value : values) {
		key += value;
		key.push_back('\0');
	}
	return key;
}

string PartitionValueToString(const Value &value) {
	return value.IsNull() ? HivePartitioning::DEFAULT_PARTITION_NAME
	                      : value.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
}

//! The directories that Glue records for different partition values, relative to the table location, or the default
//! if the partition is not registered. Evaluated once per partition by PhysicalCopyToFile.
//! Example, for PARTITIONED BY (dt, country), args=("dt=2020-01-01/country=IN", 2020-01-01, IN)
//! The default is an argument rather than a coalesce around the call: DuckDB's coalesce is an operator, not a catalog
//! function, so it cannot be bound by name from here.
void GluePartitionDirectoryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &directories = func_expr.Function().GetExtraFunctionInfo().Cast<GluePartitionDirectories>();
	auto count = args.size();
	auto writer = FlatVector::Writer<string_t>(result, count);
	for (idx_t row = 0; row < count; row++) {
		// look up known directory given partition values
		vector<string> values;
		for (idx_t col = 1; col < args.ColumnCount(); col++) {
			values.push_back(PartitionValueToString(args.GetValue(col, row)));
		}
		auto entry = directories.by_values.find(PartitionValuesKey(values));

		// unregistered, pick args[0] unless when it is null
		if (entry == directories.by_values.end()) {
			auto default_dir = args.GetValue(0, row);
			if (default_dir.IsNull()) {
				writer.WriteNull();
			} else {
				writer.WriteValue(StringVector::AddString(result, StringValue::Get(default_dir)));
			}
			continue;
		}
		if (!entry->second.refusal.empty()) {
			// this partition's registered location is known to not be expressible
			// we deferred throwing an exception until we actually have to write
			throw NotImplementedException(entry->second.refusal);
		}
		writer.WriteValue(StringVector::AddString(result, entry->second.directory));
	}
}

//! Classify each registered partition: the directory to write it to, or why DuckDB's partitioned writer will not
//! write to that location which happens due to two reasons:
//!  1/ outside the table location: PhysicalCopyToFile writes every partition directory relative to the COPY target.
//!  2/ nested inside, or equal to, another partition's location: a write cannot tell which of the two overlapping
//!     partitions a new directory belongs to. Reads can, resolving a file to the deepest registered location.
//!
//! We are deliberately stricter than Glue on writes. Reads of both layouts are unaffected.
void ClassifyPartitions(const GlueTableInfo &table_info, const string &location,
                        const vector<GluePartitionInfo> &partitions, GluePartitionDirectories &directories,
                        unordered_set<string> &registered_locations) {
	auto describe = [](const GluePartitionInfo &partition) {
		return StringUtil::Format("(%s)", StringUtil::Join(partition.values, ", "));
	};
	// a writable partition's relative directory, with every key it is indexed under (the value as Glue reports it,
	// plus its unescaped form when those differ), for the overlap check below
	vector<pair<string, vector<string>>> relative_keys;
	for (auto &partition : partitions) {
		auto partition_location = partition.location;
		StringUtil::RTrim(partition_location, "/");
		if (partition_location.empty()) {
			continue;
		}
		registered_locations.insert(partition_location);
		auto key = PartitionValuesKey(partition.values);
		// A catalog written before the values were stored unescaped holds the escaped form ('a%20b') where the data
		// says 'a b', and the write matches on the values Glue reports-- so index and accept either.
		vector<string> unescaped_values;
		for (auto &value : partition.values) {
			unescaped_values.push_back(HivePartitioning::Unescape(value));
		}
		auto unescaped_key = PartitionValuesKey(unescaped_values);
		GluePartitionDirectory entry;
		if (partition_location == location) {
			entry.refusal = StringUtil::Format(
			    "Cannot write to partition %s of Hive table '%s.%s': it is registered at the table location itself "
			    "(\"%s\"), so it has no directory of its own to write into",
			    describe(partition), table_info.database_name, table_info.name, partition_location);
		} else if (!StringUtil::StartsWith(partition_location, location + "/")) {
			entry.refusal = StringUtil::Format(
			    "Cannot write to partition %s of Hive table '%s.%s': it is registered at \"%s\", which is outside "
			    "the table location \"%s\". Glue allows that, but Duck will not. Please repoint "
			    "the partition below the table location with ALTER TABLE ... PARTITION ... SET LOCATION",
			    describe(partition), table_info.database_name, table_info.name, partition_location, location);
		} else {
			entry.directory = partition_location.substr(location.size() + 1);
			vector<string> keys {key};
			if (unescaped_key != key) {
				keys.push_back(unescaped_key);
			}
			relative_keys.emplace_back(entry.directory, std::move(keys));
		}
		if (unescaped_key != key) {
			directories.by_values[unescaped_key] = entry;
		}
		directories.by_values[key] = std::move(entry);
	}
	// no two partitions may share a directory, and none may sit inside another's. Sorted, a directory is preceded by
	// every one of its ancestors ("d" < "d/x", being a prefix and shorter), so a stack of the ancestors still open at
	// this point finds every overlap. Comparing only adjacent pairs would miss the second of two directories nested in
	// the same ancestor, e.g. d, d/x/1, d/x/2.
	std::sort(relative_keys.begin(), relative_keys.end());
	vector<idx_t> open_ancestors;
	for (idx_t i = 0; i < relative_keys.size(); i++) {
		auto &current = relative_keys[i];
		// drop the ancestors this directory is not inside of; an equal directory stays, so a later descendant of it is
		// still seen
		while (!open_ancestors.empty()) {
			auto &candidate = relative_keys[open_ancestors.back()].first;
			if (candidate == current.first || StringUtil::StartsWith(current.first, candidate + "/")) {
				break;
			}
			open_ancestors.pop_back();
		}
		open_ancestors.push_back(i);
		if (open_ancestors.size() < 2) {
			continue;
		}
		auto &previous = relative_keys[open_ancestors[open_ancestors.size() - 2]];
		bool shared = current.first == previous.first;
		auto refusal =
		    shared
		        ? StringUtil::Format("Cannot write to Hive table '%s.%s': two of its partitions are both registered at "
		                             "\"%s\". Glue allows that, but a write cannot tell which of them a file in that "
		                             "directory belongs to. Reading the table is unaffected",
		                             table_info.database_name, table_info.name, previous.first)
		        : StringUtil::Format("Cannot write to Hive table '%s.%s': its partition at \"%s\" is nested inside "
		                             "the location \"%s\" of another partition. Glue allows that, and reading "
		                             "resolves a file to the deepest registered location, but a write cannot tell "
		                             "which of the two a new directory belongs to. Reading the table is unaffected",
		                             table_info.database_name, table_info.name, current.first, previous.first);
		// both sides of the overlap are unwritable, at least one is ambiguous
		for (auto &key : previous.second) {
			directories.by_values[key].refusal = refusal;
		}
		for (auto &key : current.second) {
			directories.by_values[key].refusal = refusal;
		}
	}
}

//! Build the PARTITION_PATH expression: the location Glue records for a partition, falling back to the default hive
//! layout for a partition that is not registered yet. The default is built from DuckDB's own hive_partition_component.
unique_ptr<Expression> BuildPartitionPath(ClientContext &context, const GlueTableInfo &table_info,
                                          const string &location, const vector<Identifier> &copy_names,
                                          const vector<LogicalType> &copy_types, const vector<idx_t> &partition_columns,
                                          const vector<GluePartitionInfo> &partitions,
                                          unordered_set<string> &registered_locations) {
	auto directories = make_shared_ptr<GluePartitionDirectories>();
	ClassifyPartitions(table_info, location, partitions, *directories, registered_locations);
	if (directories->by_values.empty()) {
		// nothing registered yet, so every partition takes the default layout
		return nullptr;
	}

	FunctionBinder function_binder(context);
	auto component_function = HivePartitionComponentFun::GetFunction();
	vector<unique_ptr<Expression>> components;
	for (idx_t i = 0; i < partition_columns.size(); i++) {
		auto column_index = partition_columns[i];
		vector<unique_ptr<Expression>> children;
		children.push_back(make_uniq<BoundConstantExpression>(Value(copy_names[column_index].GetIdentifierName())));
		children.push_back(make_uniq<BoundReferenceExpression>(copy_types[column_index], i));
		components.push_back(function_binder.BindScalarFunction(component_function, std::move(children)));
	}
	auto hive_layout = function_binder.BindScalarFunction(PathJoinFun::GetFunction(), std::move(components));

	// glue_partition_directory(<hive layout>, <partition values...>): the registered directory, else that layout
	ScalarFunction lookup_function("glue_partition_directory", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                               GluePartitionDirectoryFunction);
	lookup_function.SetVarArgs(LogicalType::ANY);
	lookup_function.SetExtraFunctionInfo(std::move(directories));
	vector<unique_ptr<Expression>> lookup_children;
	lookup_children.push_back(std::move(hive_layout));
	for (idx_t i = 0; i < partition_columns.size(); i++) {
		lookup_children.push_back(make_uniq<BoundReferenceExpression>(copy_types[partition_columns[i]], i));
	}
	return function_binder.BindScalarFunction(lookup_function, std::move(lookup_children));
}

} // namespace

static optional_ptr<CopyFunctionCatalogEntry> TryGetCopyFunction(DatabaseInstance &db, const string &name) {
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	auto &schema = system_catalog.GetSchema(data, Identifier::DefaultSchema());
	auto entry = schema.GetEntry(data, CatalogType::COPY_FUNCTION_ENTRY, Identifier(name));
	if (!entry) {
		return nullptr;
	}
	return &entry->Cast<CopyFunctionCatalogEntry>();
}

PhysicalOperator &GlueHiveInsert::PlanWrite(ClientContext &context, PhysicalPlanGenerator &planner, LogicalOperator &op,
                                            GlueTable &table, PhysicalOperator &plan, const vector<Identifier> &names,
                                            const vector<LogicalType> &types) {
	// Ask Glue for the current definition: the location (and the format) may have changed since the entry was
	// created, e.g. through ALTER TABLE ... SET LOCATION
	auto table_info = table.RefreshTableInfo(context);
	auto location = table_info.location;
	StringUtil::RTrim(location, "/");

	// The partition keys of the table, by position in the rows we are given (which may be in query order for
	// CREATE TABLE AS)
	vector<idx_t> partition_columns;
	for (auto &key : table_info.partition_keys) {
		bool found = false;
		for (idx_t i = 0; i < names.size(); i++) {
			if (StringUtil::CIEquals(names[i].GetIdentifierName(), key.name)) {
				partition_columns.push_back(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InternalException("Partition key '%s' of Hive table '%s' not found in the rows to insert", key.name,
			                        table_info.name);
		}
	}

	// the files are written in the table's format (from its SerDe)
	auto file_format = table_info.GetFileFormat();
	auto format_name = HiveFileFormatToString(file_format);
	// the operator feeding the copy and the columns it produces
	optional_ptr<PhysicalOperator> source = &plan;
	vector<Identifier> copy_names = names;
	vector<LogicalType> copy_types = types;
	// the copy function and its options
	string copy_format = format_name;
	identifier_map_t<vector<Value>> copy_options;
	switch (file_format) {
	case HiveFileFormat::PARQUET:
		break;
	case HiveFileFormat::AVRO:
		ExtensionHelper::AutoLoadExtension(context, "avro");
		break;
	case HiveFileFormat::CSV:
		// Hive CSV files: the table's dialect, a header line only when the table says so
		copy_options[Identifier("header")] = {Value::BOOLEAN(table_info.HasHeader())};
		copy_options[Identifier("delimiter")] = {Value(table_info.GetFieldDelimiter())};
		copy_options[Identifier("quote")] = {Value(table_info.GetQuoteCharacter())};
		copy_options[Identifier("escape")] = {Value(table_info.GetEscapeCharacter())};
		break;
	case HiveFileFormat::JSON: {
		// DuckDB writes JSON the way COPY ... (FORMAT json) does: every row becomes one JSON object (to_json over a
		// struct of the data columns) and the objects are written line by line with the CSV writer. The partition
		// columns stay separate columns so the copy can partition on them.
		ExtensionHelper::AutoLoadExtension(context, "json");
		vector<unique_ptr<Expression>> struct_children;
		for (idx_t i = 0; i < names.size(); i++) {
			if (std::find(partition_columns.begin(), partition_columns.end(), i) != partition_columns.end()) {
				continue;
			}
			auto column_ref = make_uniq<BoundReferenceExpression>(types[i], i);
			column_ref->SetAlias(names[i]);
			struct_children.push_back(std::move(column_ref));
		}
		vector<unique_ptr<Expression>> to_json_children;
		to_json_children.push_back(StructPackFun::GetFunction().Bind(context, std::move(struct_children)));
		FunctionBinder function_binder(context);
		ErrorData error;
		auto to_json = function_binder.BindScalarFunction(Identifier::DefaultSchema(), Identifier("to_json"),
		                                                  std::move(to_json_children), error);
		if (!to_json) {
			error.Throw();
		}
		vector<unique_ptr<Expression>> select_list;
		vector<LogicalType> projected_types;
		// to_json returns the JSON type (a VARCHAR alias): keep it as is, the executor checks the exact type
		auto json_type = to_json->GetReturnType();
		select_list.push_back(std::move(to_json));
		projected_types.push_back(json_type);
		copy_names = {Identifier("json")};
		copy_types = {json_type};
		for (idx_t i = 0; i < partition_columns.size(); i++) {
			auto column_index = partition_columns[i];
			select_list.push_back(make_uniq<BoundReferenceExpression>(types[column_index], column_index));
			projected_types.push_back(types[column_index]);
			copy_names.push_back(names[column_index]);
			copy_types.push_back(types[column_index]);
			// the partition columns follow the json column
			partition_columns[i] = 1 + i;
		}
		auto &projection = planner.Make<PhysicalProjection>(std::move(projected_types), std::move(select_list),
		                                                    op.estimated_cardinality);
		projection.children.push_back(plan);
		source = &projection;
		copy_format = "csv";
		copy_options[Identifier("quote")] = {Value("")};
		copy_options[Identifier("escape")] = {Value("")};
		copy_options[Identifier("delimiter")] = {Value("\n")};
		copy_options[Identifier("header")] = {Value::BOOLEAN(false)};
		break;
	}
	}
	// Where the partitions of this table actually live. Read from Glue.
	GlueHiveWriteInfo write_info;
	write_info.table_info = table_info;
	unique_ptr<Expression> partition_path;
	if (!partition_columns.empty()) {
		auto &glue_catalog = table.catalog.Cast<GlueCatalog>();
		auto partitions = GlueAPI::GetPartitions(context, glue_catalog, table_info.database_name, table_info.name);
		partition_path = BuildPartitionPath(context, table_info, location, copy_names, copy_types, partition_columns,
		                                    partitions, write_info.registered_locations);
	}

	auto copy_function = TryGetCopyFunction(*context.db, copy_format);
	if (!copy_function) {
		throw MissingExtensionException("Writing to Hive table '%s' requires the %s copy function", table_info.name,
		                                copy_format);
	}
	auto copy_info = make_uniq<CopyInfo>();
	copy_info->file_path = location;
	copy_info->format = copy_format;
	copy_info->is_from = false;
	copy_info->options = std::move(copy_options);

	// Hive convention: partition columns live in the directory names, not in the files
	CopyFunctionBindInput bind_input(*copy_info);
	auto names_to_write = LogicalCopyToFile::GetNamesWithoutPartitions(copy_names, partition_columns, false);
	auto types_to_write = LogicalCopyToFile::GetTypesWithoutPartitions(copy_types, partition_columns, false);
	auto function_data = copy_function->function.copy_to_bind(context, bind_input, names_to_write, types_to_write);

	auto &physical_copy = planner.Make<PhysicalCopyToFile>(
	    GetCopyFunctionReturnLogicalTypes(CopyFunctionReturnType::CHANGED_ROWS_AND_FILE_LIST), copy_function->function,
	    std::move(function_data), op.estimated_cardinality);
	auto &copy = physical_copy.Cast<PhysicalCopyToFile>();
	auto write_id = UUID::ToString(UUID::GenerateRandomUUID());
	copy.use_tmp_file = false;
	if (!partition_columns.empty()) {
		copy.file_path = location;
		copy.partition_output = true;
		copy.partition_columns = partition_columns;
		copy.write_partition_columns = false;
		// a partition registered in Glue is written to the location Glue records; everything else takes the default
		// hive layout
		copy.hive_file_pattern = true;
		copy.partition_path_expression = std::move(partition_path);
		copy.filename_pattern.SetFilenamePattern("duckdb_" + write_id + "_{i}");
		// with partitioned output the copy must not initialize a single (partition-less) output file
		copy.write_empty_file = true;
	} else {
		copy.file_path = location + "/duckdb_" + write_id + "." + format_name;
		copy.partition_output = false;
		copy.write_empty_file = false;
	}
	copy.file_extension = format_name;
	copy.overwrite_mode = CopyOverwriteMode::COPY_OVERWRITE_OR_IGNORE;
	copy.per_thread_output = false;
	copy.return_type = CopyFunctionReturnType::CHANGED_ROWS_AND_FILE_LIST;
	copy.names = copy_names;
	copy.expected_types = copy_types;
	copy.children.push_back(*source);

	auto &insert = planner.Make<GlueHiveInsert>(op, table, false, std::move(write_info));
	insert.children.push_back(physical_copy);
	return insert;
}

PhysicalOperator &GlueHiveInsert::PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
                                             GlueTable &table, optional_ptr<PhysicalOperator> plan) {
	if (op.return_chunk) {
		throw BinderException("RETURNING clause not yet supported for insertion into a Hive table");
	}
	if (op.on_conflict_info.action_type != OnConflictAction::THROW) {
		throw BinderException("ON CONFLICT clause not yet supported for insertion into a Hive table");
	}
	D_ASSERT(plan);
	// bring the rows into the table's column order, filling in defaults (NULL) for unmentioned columns
	if (!op.column_index_map.empty()) {
		plan = &planner.ResolveDefaultsProjection(op, *plan);
	}
	vector<Identifier> names;
	vector<LogicalType> types;
	for (auto &column : table.GetColumns().Logical()) {
		names.push_back(column.Name());
		types.push_back(column.Type());
	}
	return PlanWrite(context, planner, op, table, *plan, names, types);
}

PhysicalOperator &GlueHiveInsert::PlanCreateTableAs(ClientContext &context, PhysicalPlanGenerator &planner,
                                                    LogicalCreateTable &op, PhysicalOperator &plan) {
	// Create the table in Glue first (Glue has no transactions, the table exists from here on even if the insert
	// fails), then write the query result into it
	auto &glue_catalog = op.schema.catalog.Cast<GlueCatalog>();
	auto transaction = glue_catalog.GetCatalogTransaction(context);
	auto entry = op.schema.CreateTable(transaction, *op.info);

	vector<Identifier> names;
	vector<LogicalType> types;
	for (auto &column : op.info->Base().columns.Logical()) {
		names.push_back(column.Name());
		types.push_back(column.Type());
	}
	if (!entry) {
		// CREATE TABLE IF NOT EXISTS on an existing table: nothing is created and nothing is inserted
		auto &base = op.info->Base();
		EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, QualifiedName(base.GetTableName()));
		auto existing = op.schema.LookupEntry(transaction, lookup);
		auto &insert = planner.Make<GlueHiveInsert>(op, existing->Cast<GlueTable>(), true);
		insert.children.push_back(plan);
		return insert;
	}
	return PlanWrite(context, planner, op, entry->Cast<GlueTable>(), plan, names, types);
}

//===--------------------------------------------------------------------===//
// Execution
//===--------------------------------------------------------------------===//
struct GlueHiveInsertGlobalState : public GlobalSinkState {
	mutex lock;
	idx_t insert_count = 0;
	vector<string> written_files;
};

unique_ptr<GlobalSinkState> GlueHiveInsert::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<GlueHiveInsertGlobalState>();
}

SinkResultType GlueHiveInsert::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<GlueHiveInsertGlobalState>();
	lock_guard<mutex> guard(state.lock);
	if (discard) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	// the COPY reports (rows written, files written)
	for (idx_t r = 0; r < chunk.size(); r++) {
		state.insert_count += chunk.GetValue(0, r).GetValue<idx_t>();
		auto files = chunk.GetValue(1, r);
		if (files.IsNull()) {
			continue;
		}
		for (auto &file : ListValue::GetChildren(files)) {
			state.written_files.push_back(file.GetValue<string>());
		}
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType GlueHiveInsert::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<GlueHiveInsertGlobalState>();
	// The table as of plan time because the location or the partition keys may have changed since the catalog
	// entry was built.
	auto &table_info = write_info.table_info.name.empty() ? table.table_info : write_info.table_info;
	if (discard || table_info.partition_keys.empty() || state.written_files.empty()) {
		return SinkFinalizeType::READY;
	}

	// Register the partition directories the files were written to. New partitions took the default hive layout, so
	// their directory names are <key>=<value> in partition key order. Files written into an already-registered
	// location are skipped.
	auto location = table_info.location;
	StringUtil::RTrim(location, "/");
	// note: we use string here because the key is an S3 path and S3 is case-sensitive
	unordered_map<string, GluePartitionInput> partitions;
	for (auto &file : state.written_files) {
		auto directory = file.substr(0, file.find_last_of('/'));
		if (partitions.find(directory) != partitions.end()) {
			continue;
		}
		if (write_info.registered_locations.find(directory) != write_info.registered_locations.end()) {
			continue;
		}
		auto parsed = HivePartitioning::Parse(file);
		GluePartitionInput partition;
		partition.location = directory;
		for (auto &key : table_info.partition_keys) {
			auto value = parsed.find(key.name);
			if (value == parsed.end()) {
				throw InternalException("Written file '%s' has no value for partition key '%s'", file, key.name);
			}
			// Parse is a path parser and deliberately does not unescape
			partition.values.push_back(HivePartitioning::Unescape(value->second));
		}
		partitions.emplace(directory, std::move(partition));
	}
	if (partitions.empty()) {
		return SinkFinalizeType::READY;
	}
	vector<GluePartitionInput> to_register;
	for (auto &entry : partitions) {
		to_register.push_back(entry.second);
	}
	auto &glue_catalog = table.catalog.Cast<GlueCatalog>();
	GlueAPI::BatchCreatePartitions(context, glue_catalog, table_info.database_name, table_info.name, to_register);
	return SinkFinalizeType::READY;
}

SourceResultType GlueHiveInsert::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &state = sink_state->Cast<GlueHiveInsertGlobalState>();
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(state.insert_count)));
	return SourceResultType::FINISHED;
}

string GlueHiveInsert::GetName() const {
	return "GLUE_HIVE_INSERT";
}

InsertionOrderPreservingMap<string> GlueHiveInsert::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Table Name"] = table.name.GetIdentifierName();
	return result;
}

} // namespace duckdb
