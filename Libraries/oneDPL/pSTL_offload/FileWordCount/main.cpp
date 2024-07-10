//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

// Counting Words in Files Example
// by Bartlomiej Filipek, bfilipek.com
// based on C++17 Complete by N. Jossutis
// also on an example from Bryce Lelbach's talk about parallel algorithms:
// The C++17 Parallel Algorithms Library and Beyond

#include <cctype>
#include <chrono>
#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>

void PrintTiming(
    const char* title,
    const std::chrono::time_point<std::chrono::steady_clock>& start) {
  const auto end = std::chrono::steady_clock::now();
  std::cout << title << ": "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms\n";
}

// General template function for Gathering All Text files from the root
// directory uses recursive directory iterator
template <typename Policy>
std::vector<std::filesystem::path> GatherAllTextFiles(
    const std::filesystem::path& root, Policy pol) {
  std::vector<std::filesystem::path> paths;
  std::vector<std::filesystem::path> output;

  try {
    auto start = std::chrono::steady_clock::now();

    std::filesystem::recursive_directory_iterator dirpos{root};

    // for now std::copy only works with random access iterators, so filesystem
    // iterator will be one-thread only
    std::copy(begin(dirpos), end(dirpos), std::back_inserter(paths));

    PrintTiming("gathering all the paths", start);

    std::cout << "number of all files: " << std::size(paths) << "\n";

    std::mutex mut;  // we need some blocking mechanism for the output...

    start = std::chrono::steady_clock::now();

    // we have all files now... so filter them out (possibly in a parallel way,
    // as std::copy_if is not there yet
    std::for_each(
        pol, std::begin(paths), std::end(paths),
        [&output, &mut](const std::filesystem::path& p) {
          if (std::filesystem::is_regular_file(p) && p.has_extension()) {
            auto ext = p.extension();
            if (ext == std::string(".txt")) {
              std::unique_lock<std::mutex> lock(mut);
              output.push_back(p);
            }
          }
        });

    PrintTiming("filtering only TXT files", start);
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return {};
  }

  return output;
}

// Sequential version, uses copy_if
template <>
std::vector<std::filesystem::path>
GatherAllTextFiles<std::execution::sequenced_policy>(
    const std::filesystem::path& root, std::execution::sequenced_policy) {
  std::vector<std::filesystem::path> paths;
  try {
    auto start = std::chrono::steady_clock::now();

    std::filesystem::recursive_directory_iterator dirpos{root};

    std::copy_if(
        begin(dirpos), end(dirpos), std::back_inserter(paths),
        [](const std::filesystem::path& p) {
          if (std::filesystem::is_regular_file(p) && p.has_extension()) {
            auto ext = p.extension();
            return ext == std::string(".txt");
          }

          return false;
        });
    PrintTiming("filtering only TXT files sequential", start);
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return {};
  }

  return paths;
}

struct FileAndWordCount {
  std::filesystem::path path;
  uint32_t wordCount;
};

int CountWords(std::istream& in) {
  int count = 0;
  for (std::string word; in >> word; ++count) {
  }
  return count;
}

bool IsWordBeginning(char left, char right) {
  return std::isspace(left) && !std::isspace(right);
}

template <typename Policy>
std::size_t CountWords(std::string_view s, Policy policy) {
  if (s.empty()) return 0;

  std::size_t wc = (!std::isspace(s.front()) ? 1 : 0);
  wc += std::transform_reduce(policy, s.begin(), s.end() - 1, s.begin() + 1,
                              std::size_t(0), std::plus<std::size_t>(),
                              IsWordBeginning);

  return wc;
}

std::string GetFileContents(const std::filesystem::path& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (in) {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(static_cast<size_t>(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return (contents);
  }

  return "";
}

template <typename Policy>
uintmax_t CountWordsInFiles(Policy pol,
                            std::vector<FileAndWordCount>& filesWithWordCount) {
  uintmax_t allFilesWordCount = 0;

  allFilesWordCount = std::transform_reduce(
      pol, filesWithWordCount.begin(), filesWithWordCount.end(),
      std::uintmax_t{0}, std::plus<>(), [](FileAndWordCount& p) {
        const auto str = GetFileContents(p.path);
        p.wordCount = CountWords(str, std::execution::par_unseq);
        return p.wordCount;
      });

  return allFilesWordCount;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0]
              << " <path> <parallel:1|0> <showcount:1:0>\n";
    return EXIT_FAILURE;
  }
  std::filesystem::path root{argv[1]};

  int executionPolicyMode = atoi(argv[2]);
  std::cout << "Using " << (executionPolicyMode ? "PAR" : "SEQ") << " Policy\n";

  std::vector<std::filesystem::path> paths;

  if (executionPolicyMode)
    paths = GatherAllTextFiles(root, std::execution::par_unseq);
  else
    paths = GatherAllTextFiles(root, std::execution::seq);

  std::cout << "number of files: " << std::size(paths) << "\n";

  // transform into pairs:
  std::vector<FileAndWordCount> filesWithWordCount;
  std::transform(std::begin(paths), std::end(paths),
                 std::back_inserter(filesWithWordCount),
                 [](const std::filesystem::path& p) {
                   return FileAndWordCount{p, 0};
                 });

  // accumulate size of all regular files:
  auto start = std::chrono::steady_clock::now();
  uintmax_t allWordsCount = 0;
  if (executionPolicyMode)
    allWordsCount = CountWordsInFiles(std::execution::par_unseq, filesWithWordCount);
  else
    allWordsCount = CountWordsInFiles(std::execution::seq, filesWithWordCount);

  PrintTiming("computing the sizes", start);
  std::cout << "word count of " << paths.size()
            << " TXT files: " << allWordsCount << "\n";

  if (argc > 3) {
    for (const auto& p : filesWithWordCount)
      std::cout << p.path << ", words: " << p.wordCount << "\n";
  }
}
