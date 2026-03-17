# Project Rules

- Always run `bazel run //:refresh_compile_commands` when new header or source files are added.
- When a function needs to store a string into a member, use the sink parameter pattern: take `std::string` by value and `std::move()` into the member.
- For `std::unordered_map`: use `at()` for lookup (throws on missing key, works on const), never `operator[]` (silently inserts). Use `emplace()` for insertion.
