// CUDA at Scale Independent Project
// Batch Gaussian Blur using CUDA NPP library
//
// Processes all .pgm images in a given input directory,
// applies Gaussian blur on the GPU, and writes results to output directory.
//
// Usage:
//   ./bin/image_processor --input <input_dir> --output <output_dir> [--sigma <value>]
//
// Google C++ Style Guide compliant.

#include <npp.h>
#include <nppi_filtering_functions.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dirent.h>
  #include <sys/stat.h>
#endif

// ----------------------------------------------------------------------------
// Helpers: PGM I/O (P5 binary grayscale)
// ----------------------------------------------------------------------------

struct PgmImage {
  int width;
  int height;
  int max_val;
  std::vector<unsigned char> pixels;  // row-major, 1 byte per pixel
};

static bool LoadPgm(const std::string& path, PgmImage* img) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open input file: %s\n", path.c_str());
    return false;
  }

  char magic[3];
  if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
    fprintf(stderr, "[ERROR] Not a binary PGM (P5) file: %s\n", path.c_str());
    fclose(fp);
    return false;
  }

  // Skip comments
  int c = fgetc(fp);
  while (c == '#') {
    while (fgetc(fp) != '\n') {}
    c = fgetc(fp);
  }
  ungetc(c, fp);

  if (fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max_val) != 3) {
    fprintf(stderr, "[ERROR] Malformed PGM header: %s\n", path.c_str());
    fclose(fp);
    return false;
  }
  fgetc(fp);  // consume single whitespace after header

  size_t npixels = static_cast<size_t>(img->width) * img->height;
  img->pixels.resize(npixels);
  if (fread(img->pixels.data(), 1, npixels, fp) != npixels) {
    fprintf(stderr, "[ERROR] Truncated pixel data: %s\n", path.c_str());
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

static bool SavePgm(const std::string& path, const PgmImage& img) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open output file: %s\n", path.c_str());
    return false;
  }
  fprintf(fp, "P5\n%d %d\n%d\n", img.width, img.height, img.max_val);
  size_t npixels = static_cast<size_t>(img.width) * img.height;
  fwrite(img.pixels.data(), 1, npixels, fp);
  fclose(fp);
  return true;
}

// ----------------------------------------------------------------------------
// Helpers: directory listing
// ----------------------------------------------------------------------------

static std::vector<std::string> ListPgmFiles(const std::string& dir) {
  std::vector<std::string> files;

#ifdef _WIN32
  std::string pattern = dir + "\\*.pgm";
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return files;
  do {
    files.push_back(dir + "\\" + fd.cFileName);
  } while (FindNextFileA(h, &fd));
  FindClose(h);
#else
  DIR* d = opendir(dir.c_str());
  if (!d) return files;
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    std::string name(entry->d_name);
    if (name.size() > 4 &&
        name.substr(name.size() - 4) == ".pgm") {
      files.push_back(dir + "/" + name);
    }
  }
  closedir(d);
#endif

  return files;
}

static std::string Basename(const std::string& path) {
#ifdef _WIN32
  size_t pos = path.find_last_of("/\\");
#else
  size_t pos = path.find_last_of('/');
#endif
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// ----------------------------------------------------------------------------
// GPU: Apply Gaussian blur using NPP
// ----------------------------------------------------------------------------

// Returns false on any CUDA or NPP error.
static bool ApplyGaussianBlur(const PgmImage& src, PgmImage* dst,
                               int mask_size) {
  // mask_size must be 3, 5, 7, 9, or 11 for nppiFilterGauss_8u_C1R
  if (mask_size != 3 && mask_size != 5 && mask_size != 7 &&
      mask_size != 9  && mask_size != 11) {
    fprintf(stderr, "[ERROR] mask_size must be 3/5/7/9/11, got %d\n",
            mask_size);
    return false;
  }

  int w = src.width;
  int h = src.height;
  size_t npixels = static_cast<size_t>(w) * h;

  // Allocate device memory
  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;
  int src_step = 0;
  int dst_step = 0;

  d_src = nppiMalloc_8u_C1(w, h, &src_step);
  d_dst = nppiMalloc_8u_C1(w, h, &dst_step);
  if (!d_src || !d_dst) {
    fprintf(stderr, "[ERROR] nppiMalloc failed\n");
    nppiFree(d_src);
    nppiFree(d_dst);
    return false;
  }

  // Upload source pixels (host -> device)
  // nppiMalloc may add padding; copy row by row to respect pitch.
  for (int row = 0; row < h; ++row) {
    cudaMemcpy(d_src + row * src_step,
               src.pixels.data() + row * w,
               w,
               cudaMemcpyHostToDevice);
  }

  // Run NPP Gaussian filter
  NppiSize roi = {w, h};
  NppMaskSize npp_mask;
  switch (mask_size) {
    case 3:  npp_mask = NPP_MASK_SIZE_3_X_3;  break;
    case 5:  npp_mask = NPP_MASK_SIZE_5_X_5;  break;
    case 7:  npp_mask = NPP_MASK_SIZE_7_X_7;  break;
    case 9:  npp_mask = NPP_MASK_SIZE_9_X_9;  break;
    default: npp_mask = NPP_MASK_SIZE_11_X_11; break;
  }

  NppStatus status = nppiFilterGauss_8u_C1R(
      d_src, src_step,
      d_dst, dst_step,
      roi, npp_mask);

  if (status != NPP_SUCCESS) {
    fprintf(stderr, "[ERROR] nppiFilterGauss_8u_C1R failed with code %d\n",
            static_cast<int>(status));
    nppiFree(d_src);
    nppiFree(d_dst);
    return false;
  }

  // Download result (device -> host)
  dst->width   = w;
  dst->height  = h;
  dst->max_val = src.max_val;
  dst->pixels.resize(npixels);

  for (int row = 0; row < h; ++row) {
    cudaMemcpy(dst->pixels.data() + row * w,
               d_dst + row * dst_step,
               w,
               cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize();
  nppiFree(d_src);
  nppiFree(d_dst);
  return true;
}

// ----------------------------------------------------------------------------
// CLI argument parsing
// ----------------------------------------------------------------------------

struct Config {
  std::string input_dir;
  std::string output_dir;
  int mask_size;

  Config() : mask_size(5) {}
};

static void PrintUsage(const char* prog) {
  fprintf(stderr,
    "Usage: %s --input <input_dir> --output <output_dir> [--mask <3|5|7|9|11>]\n"
    "\n"
    "  --input   Directory containing .pgm images\n"
    "  --output  Directory to write blurred .pgm images\n"
    "  --mask    Gaussian mask size (default: 5)\n"
    "\n"
    "Example:\n"
    "  %s --input data/input --output data/output --mask 7\n",
    prog, prog);
}

static bool ParseArgs(int argc, char** argv, Config* cfg) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      cfg->input_dir = argv[++i];
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      cfg->output_dir = argv[++i];
    } else if (strcmp(argv[i], "--mask") == 0 && i + 1 < argc) {
      cfg->mask_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0 ||
               strcmp(argv[i], "-h") == 0) {
      PrintUsage(argv[0]);
      exit(0);
    } else {
      fprintf(stderr, "[ERROR] Unknown argument: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (cfg->input_dir.empty() || cfg->output_dir.empty()) {
    fprintf(stderr, "[ERROR] --input and --output are required.\n");
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg;
  if (!ParseArgs(argc, argv, &cfg)) return 1;

  // Print CUDA device info
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "[ERROR] No CUDA-capable GPU found.\n");
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("[INFO] GPU: %s (CUDA %d.%d)\n", prop.name,
         prop.major, prop.minor);
  printf("[INFO] Input dir  : %s\n", cfg.input_dir.c_str());
  printf("[INFO] Output dir : %s\n", cfg.output_dir.c_str());
  printf("[INFO] Mask size  : %dx%d\n", cfg.mask_size, cfg.mask_size);
  printf("[INFO] -------------------------------------------\n");

  // Collect input files
  std::vector<std::string> files = ListPgmFiles(cfg.input_dir);
  if (files.empty()) {
    fprintf(stderr, "[ERROR] No .pgm files found in: %s\n",
            cfg.input_dir.c_str());
    return 1;
  }
  printf("[INFO] Found %zu image(s) to process.\n", files.size());

  int success_count = 0;
  int fail_count    = 0;

  auto wall_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < files.size(); ++i) {
    const std::string& in_path = files[i];
    std::string fname   = Basename(in_path);
    std::string out_path = cfg.output_dir + "/" + fname;

    auto t0 = std::chrono::steady_clock::now();

    PgmImage src;
    if (!LoadPgm(in_path, &src)) {
      ++fail_count;
      continue;
    }

    PgmImage dst;
    if (!ApplyGaussianBlur(src, &dst, cfg.mask_size)) {
      ++fail_count;
      continue;
    }

    if (!SavePgm(out_path, dst)) {
      ++fail_count;
      continue;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("[%3zu/%3zu] %-30s  %4dx%4d  %.2f ms\n",
           i + 1, files.size(), fname.c_str(),
           src.width, src.height, ms);
    ++success_count;
  }

  auto wall_end = std::chrono::steady_clock::now();
  double wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  printf("[INFO] -------------------------------------------\n");
  printf("[INFO] Done. %d succeeded, %d failed. Total wall time: %.2f ms\n",
         success_count, fail_count, wall_ms);

  return (fail_count > 0) ? 1 : 0;
}
