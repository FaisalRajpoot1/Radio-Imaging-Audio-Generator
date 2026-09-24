# Raw measurement data

These files are the raw output behind every number in the main README. They were measured on 24 September 2026. Local folder paths are shortened to `<scratch>` (a temporary folder) and `<repo>` (this repository); no measured value was changed.

**Laptop:** Intel Core i7-8665U (4 cores, 8 threads), 16 GB RAM, no GPU, Windows 11, Python 3.11.
**Settings:** the same prompt for every run, the MusicGen model already on disk, the network off.
**Units:** memory is in MiB, from `psutil` (`rss_mb`, `peak_mb`); install sizes are in bytes.

## Round 1: speed and reliability

| File | What it holds | README table |
|---|---|---|
| `round1_install_sizes.txt` | Size of `site-packages` without TensorFlow (1,508,523,320 bytes) and with it (2,718,850,117 bytes) | Install size |
| `round1_packages_without_tensorflow.txt`, `round1_packages_with_tensorflow.txt` | The installed packages, which show what TensorFlow added | Install size |
| `round1_clicks_run1.txt` | First click run: original app, and the fork **with TensorFlow still installed** (not used in the tables, see below) | — |
| `round1_clicks_runs2_3.jsonl` | Click runs 2 and 3: the original (with TensorFlow) against the fork (clean install), alternating | Wait per click, peak memory |
| `round1_model_load_times.txt` | Three timed model loads: 9.97, 11.3 and 11.07 s | "11 s model reload" |
| `round1_first_page_load.jsonl` | Five first page loads of each version | First page load, memory after load |

Run 1 of the fork is left out of the tables because TensorFlow was still installed, and `transformers` imports it at start-up. The fair comparison is the fork in a clean install, which runs 2 and 3 use. The original's run 1 is used, because it had its normal setup.

## Round 2: new features

| File | What it holds | README table |
|---|---|---|
| `round2_clip_length.jsonl` | Clicks for 5 s, 10 s and 20 s clips | Wait per click by clip length |
| `round2_loudness_peak_cap_only.jsonl` | 8 real clips, finished by the first version (peak cap only) | Loudness, first try |
| `round2_loudness_with_limiter.jsonl` | The same 8 clips (same seeds), finished with the look-ahead limiter | Loudness, with the limiter |

## Round 3: the free GPU Space

| File | What it holds | README table |
|---|---|---|
| `round3_live_gpu_space.jsonl` | Calls to the live Space: a warm-up, five 10 s clips, three takes, a 30 s clip, and one downloaded take's loudness | GPU timings |

`generated_s` is measured inside the app around the GPU call, so it includes waiting for ZeroGPU to hand out a GPU. Clip lengths come from exact WAV sizes: float32 mono has 4 bytes per sample plus a 58-byte header.

### Download checks (copied from the console output)

These explain the "slow downloads" finding. Times are in seconds; each row is one download from the laptop.

**A. Space files against the main Hugging Face site** (with a login header):

| Round | Space player WAV (640 KB) | Space take WAV (1.3 MB) | huggingface.co file (2.4 MB) |
|---|---|---|---|
| 1 | 53.9 | 106.2 | 1.0 |
| 2 | 2.2 | 113.8 | 1.6 |
| 3 | 56.5 | 2.7 | 1.0 |
| 4 | 48.2 | 115.4 | 1.1 |

**B. No login header**, player WAV only: 2.2, 2.0, 2.3, 53.8, 50.7, 2.4.

**C. Gradio's SSR turned off**: player WAV 2.2, 65.5, 61.7, 50.8; take WAV 111.7, 94.3, 2.8, 2.6. SSR was not the cause, so the change was undone.

**D. Gradio's own page file** (134,191 bytes), with `curl`: speeds of 11,939, 14,641, 53,262, 60,479, 13,517 and 11,527 bytes/s. The first byte always arrived in 0.72 to 1.61 s, so the server answered quickly and the transfer itself was slow.

Conclusion: the slow downloads came from the network route between this laptop and the Space's servers, not from the app. The players now load the take's MP3 (241 KB for 10 s) instead of a 640 KB WAV.
