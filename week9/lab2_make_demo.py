"""
Week 9 - Lab 2: Interactive Audio Showcase Demo Page                run:  python lab2_make_demo.py

Numerical metrics (PESQ, STOI, SI-SDR) cannot replace subjective listening.
When presenting your Master's thesis work to Prof. Hung, committee members, or conference reviewers,
having an interactive, self-contained HTML audio demonstration page is the most convincing way
to showcase speech enhancement quality.

In this lab, you:
  1. Implement waveform envelope extraction for visual display
  2. Generate pure SVG waveform plots embedded directly in HTML (zero external dependencies)
  3. Build a standalone interactive audio showcase (demo/index.html) with HTML5 <audio> players
  4. Ensure the demo is 100% self-contained and functions fully offline
"""
import html
import numpy as np
from pathlib import Path

_passed = []


def check(name, fn, hint=""):
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


# ---------------------------------------------------------------- Exercise 2.1
def downsample_waveform_envelope(waveform, num_points=100):
    """Downsample audio waveform into peak envelope points for visual rendering.

    Splits waveform into `num_points` chunks, computing (min_val, max_val) for each chunk.

    Args:
      waveform: 1D numpy array of audio samples
      num_points: number of visual segments (e.g. 100)

    Returns:
      list of (min_val, max_val) tuples, each bounded within [-1.0, 1.0]
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def generate_svg_waveform(envelope_points, width=400, height=80, color="#2563eb"):
    """Generate an inline SVG waveform string from envelope points.

    Args:
      envelope_points: list of (min_val, max_val) tuples
      width: SVG viewbox width
      height: SVG viewbox height
      color: hex color string for waveform bars

    Returns:
      svg_string: valid XML/SVG element string
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def build_html_demo(samples_list, out_html_path, title="GTCRN Speech Enhancement Showcase"):
    """Build a standalone, responsive HTML5 audio comparison demo page.

    samples_list format:
      [
        {
          "id": "p232_001",
          "noise_type": "Cafe Babble",
          "snr": "0 dB",
          "noisy_audio": "audio/p232_001_noisy.wav",
          "enh_audio": "audio/p232_001_enh.wav",
          "clean_audio": "audio/p232_001_clean.wav",
          "noisy_svg": "<svg ...>",
          "enh_svg": "<svg ...>",
          "clean_svg": "<svg ...>"
        }, ...
      ]
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    print("--- Testing Exercise 2.1: downsample_waveform_envelope ---")
    dummy_wav = np.sin(np.linspace(0, 100 * np.pi, 16000))
    try:
        env = downsample_waveform_envelope(dummy_wav, num_points=50)
    except NotImplementedError:
        env = []

    check("downsamples waveform into 50 envelope tuples",
          lambda: len(env) == 50 and isinstance(env[0], tuple),
          "expected 50 (min, max) envelope tuples")
    check("envelope points are normalized within [-1.0, 1.0]",
          lambda: len(env) > 0 and all(-1.01 <= p[0] <= p[1] <= 1.01 for p in env),
          "all points must be in range [-1.0, 1.0]")

    print("\n--- Testing Exercise 2.2: generate_svg_waveform ---")
    try:
        svg_str = generate_svg_waveform(env, width=300, height=60, color="#38bdf8")
    except NotImplementedError:
        svg_str = ""

    check("generates valid SVG container with rect elements",
          lambda: svg_str.startswith("<svg") and "</svg>" in svg_str and "<rect" in svg_str,
          "expected valid SVG markup containing rect bars")

    print("\n--- Testing Exercise 2.3: build_html_demo ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_page = Path(tmp) / "demo" / "index.html"
        test_samples = [
            {
                "id": "test_sample_01",
                "noise_type": "Babble",
                "snr": "0 dB",
                "noisy_audio": "audio/noisy_01.wav",
                "enh_audio": "audio/enh_01.wav",
                "clean_audio": "audio/clean_01.wav",
                "noisy_svg": svg_str,
                "enh_svg": svg_str,
                "clean_svg": svg_str,
            }
        ]
        try:
            demo_html = build_html_demo(test_samples, out_page)
            has_html = out_page.exists() and "<audio" in out_page.read_text(encoding="utf-8")
        except NotImplementedError:
            demo_html, has_html = "", False

        check("builds complete HTML demo with audio controls and sample cards",
              lambda: has_html and "test_sample_01" in demo_html and "<audio controls" in demo_html,
              "generated HTML must contain audio tags and sample ID")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
