#!/usr/bin/env python3
"""Generate educational wavelet tutorial showing basic functions and operations.

Usage:
    ./scripts/wavelet-tutorial.py
"""

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywt

REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Dark theme colors
COLORS = {
    'bg': '#1a1a2e',
    'card': '#16213e',
    'accent': '#0f3460',
    'highlight': '#e94560',
    'text': '#eaeaea',
    'muted': '#a0a0a0',
    'success': '#4ecca3',
    'warning': '#ffc93c',
    'info': '#00adb5',
}


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=COLORS['card'], edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str


def setup_plot_style():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['card'],
        'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['muted'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['accent'],
        'grid.alpha': 0.3,
    })


def compute_cwt(signal: np.ndarray, scales: np.ndarray = None) -> tuple:
    """Compute continuous wavelet transform."""
    if scales is None:
        scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(signal, scales, 'morl')
    power = np.abs(coeffs) ** 2
    return power, scales


def generate_signals(n_points: int = 1000, n_periods: float = 10):
    """Generate basic test signals."""
    t = np.linspace(0, n_periods * 2 * np.pi, n_points)

    signals = {
        'sine': {
            'data': np.sin(t),
            'description': 'Pure sine wave - single frequency',
            'color': COLORS['info']
        },
        'square': {
            'data': np.sign(np.sin(t)),
            'description': 'Square wave - fundamental + odd harmonics',
            'color': COLORS['highlight']
        },
        'sawtooth': {
            'data': 2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi))),
            'description': 'Sawtooth - fundamental + all harmonics',
            'color': COLORS['success']
        },
        'triangle': {
            'data': 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1,
            'description': 'Triangle wave - fundamental + odd harmonics (fast decay)',
            'color': COLORS['warning']
        },
    }

    return t, signals


def generate_frequency_signals(n_points: int = 1000):
    """Generate signals with different frequencies."""
    t = np.linspace(0, 20 * np.pi, n_points)

    signals = {
        'slow_sine': {
            'data': np.sin(t * 0.5),
            'description': 'Slow sine (period = 4π)',
            'color': COLORS['info']
        },
        'fast_sine': {
            'data': np.sin(t * 3),
            'description': 'Fast sine (period = 2π/3)',
            'color': COLORS['highlight']
        },
        'chirp': {
            'data': np.sin(t * (1 + t / 20)),
            'description': 'Chirp - frequency increases over time',
            'color': COLORS['success']
        },
    }

    return t, signals


def generate_compound_signals(n_points: int = 1000):
    """Generate compound signals from operations."""
    t = np.linspace(0, 10 * 2 * np.pi, n_points)

    sine_slow = np.sin(t * 0.5)
    sine_fast = np.sin(t * 3)

    signals = {
        'sine_slow': {
            'data': sine_slow,
            'description': 'Slow sine (base)',
            'color': COLORS['info']
        },
        'sine_fast': {
            'data': sine_fast,
            'description': 'Fast sine (base)',
            'color': COLORS['highlight']
        },
        'addition': {
            'data': sine_slow + sine_fast,
            'description': 'Addition: slow + fast',
            'color': COLORS['success']
        },
        'multiplication': {
            'data': sine_slow * sine_fast,
            'description': 'Multiplication: slow × fast (AM modulation)',
            'color': COLORS['warning']
        },
        'division': {
            'data': sine_fast / (np.abs(sine_slow) + 0.1),  # Avoid div by zero
            'description': 'Division: fast ÷ |slow| (amplitude modulation)',
            'color': '#9b59b6'
        },
    }

    return t, signals


def generate_transient_signals(n_points: int = 1000):
    """Generate signals with transient events."""
    t = np.linspace(0, 10, n_points)

    # Spike
    spike = np.zeros(n_points)
    spike[n_points // 2] = 10

    # Step
    step = np.zeros(n_points)
    step[n_points // 2:] = 1

    # Gaussian pulse
    gaussian = np.exp(-((t - 5) ** 2) / 0.1)

    # Damped oscillation
    damped = np.exp(-t) * np.sin(10 * t)
    damped = np.roll(damped, n_points // 4)

    signals = {
        'spike': {
            'data': spike,
            'description': 'Delta spike - energy at all frequencies',
            'color': COLORS['highlight']
        },
        'step': {
            'data': step,
            'description': 'Step function - low frequency content',
            'color': COLORS['info']
        },
        'gaussian': {
            'data': gaussian,
            'description': 'Gaussian pulse - localized in both domains',
            'color': COLORS['success']
        },
        'damped': {
            'data': damped,
            'description': 'Damped oscillation - frequency persists, amplitude decays',
            'color': COLORS['warning']
        },
    }

    return t, signals


def plot_signal_and_wavelet(t, signal_data, title, color, scales=None):
    """Plot time series and wavelet scalogram side by side."""
    if scales is None:
        scales = np.arange(1, 100)

    power, scales = compute_cwt(signal_data, scales)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Time domain
    ax1 = axes[0]
    ax1.plot(t, signal_data, color=color, linewidth=1.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain', fontsize=11, color=COLORS['text'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t[0], t[-1])

    # Wavelet domain
    ax2 = axes[1]
    im = ax2.imshow(np.log10(power + 1e-10), aspect='auto', cmap='magma',
                    extent=[0, len(t), scales[-1], scales[0]])
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Scale (≈ period)')
    ax2.set_title('Wavelet Power Spectrum', fontsize=11, color=COLORS['text'])

    plt.suptitle(title, fontsize=12, fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_comparison_grid(t, signals, title, scales=None):
    """Plot multiple signals in a comparison grid."""
    if scales is None:
        scales = np.arange(1, 100)

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(12, 2.5 * n_signals))

    for i, (name, sig) in enumerate(signals.items()):
        data = sig['data']
        color = sig['color']
        desc = sig['description']

        power, _ = compute_cwt(data, scales)

        # Time domain
        ax1 = axes[i, 0]
        ax1.plot(t, data, color=color, linewidth=1.2)
        ax1.set_ylabel('Amplitude', fontsize=9)
        ax1.set_title(f'{name}: {desc}', fontsize=10, color=COLORS['text'], loc='left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(t[0], t[-1])
        if i == n_signals - 1:
            ax1.set_xlabel('Time')

        # Wavelet domain
        ax2 = axes[i, 1]
        ax2.imshow(np.log10(power + 1e-10), aspect='auto', cmap='magma',
                   extent=[0, len(t), scales[-1], scales[0]])
        ax2.set_ylabel('Scale', fontsize=9)
        if i == n_signals - 1:
            ax2.set_xlabel('Time (samples)')

    plt.suptitle(title, fontsize=14, fontweight='bold', color=COLORS['text'], y=1.01)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_operation_demo(t, signals, title):
    """Plot signals showing operation effects."""
    scales = np.arange(1, 80)

    # We'll show: input1, input2, result
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))

    signal_order = ['sine_slow', 'sine_fast', 'addition', 'multiplication', 'division']

    for i, name in enumerate(signal_order):
        sig = signals[name]
        data = sig['data']
        color = sig['color']
        desc = sig['description']

        power, _ = compute_cwt(data, scales)

        # Time domain
        ax1 = axes[i, 0]
        ax1.plot(t, data, color=color, linewidth=1.2)
        ax1.set_ylabel('Amp', fontsize=9)
        ax1.set_title(desc, fontsize=10, color=COLORS['text'], loc='left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(t[0], t[-1])

        # Add separator line before operations
        if i == 2:
            ax1.axhline(y=ax1.get_ylim()[1], color=COLORS['success'], linewidth=3)

        # Wavelet domain
        ax2 = axes[i, 1]
        ax2.imshow(np.log10(power + 1e-10), aspect='auto', cmap='magma',
                   extent=[0, len(t), scales[-1], scales[0]])
        ax2.set_ylabel('Scale', fontsize=9)

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time (samples)')

    plt.suptitle(title, fontsize=14, fontweight='bold', color=COLORS['text'], y=1.01)
    plt.tight_layout()

    return fig_to_base64(fig)


def generate_html(plots: dict) -> str:
    """Generate the HTML report."""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wavelet Transform Tutorial</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: {COLORS['bg']};
            color: {COLORS['text']};
            line-height: 1.7;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid {COLORS['accent']};
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, {COLORS['info']}, {COLORS['success']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: {COLORS['muted']}; font-size: 1.1rem; }}
        .timestamp {{ color: {COLORS['muted']}; font-size: 0.85rem; margin-top: 0.5rem; }}

        .section {{
            background: {COLORS['card']};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLORS['info']};
        }}
        .section h2 {{
            font-size: 1.4rem;
            margin-bottom: 1rem;
            color: {COLORS['text']};
        }}
        .section h3 {{
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem 0;
            color: {COLORS['info']};
        }}
        .section p {{ color: {COLORS['muted']}; margin-bottom: 1rem; }}
        .section.highlight {{ border-left-color: {COLORS['highlight']}; }}
        .section.success {{ border-left-color: {COLORS['success']}; }}
        .section.warning {{ border-left-color: {COLORS['warning']}; }}

        .chart {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}

        .key-insight {{
            background: {COLORS['bg']};
            border-left: 3px solid {COLORS['success']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }}
        .key-insight strong {{
            color: {COLORS['success']};
        }}

        .formula {{
            background: {COLORS['bg']};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-family: 'Monaco', 'Consolas', monospace;
            margin: 0.5rem 0;
            display: inline-block;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid {COLORS['accent']};
        }}
        th {{
            color: {COLORS['muted']};
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}

        ul {{ margin-left: 1.5rem; margin-bottom: 1rem; }}
        li {{ margin-bottom: 0.5rem; }}

        footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['accent']};
            color: {COLORS['muted']};
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Wavelet Transform Tutorial</h1>
            <p class="subtitle">Understanding Time-Frequency Analysis Through Basic Functions</p>
            <p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="section">
            <h2>What is a Wavelet Transform?</h2>
            <p>A wavelet transform decomposes a signal into components at different <strong>scales</strong> (similar to frequencies)
               while preserving <strong>time localization</strong>. Unlike Fourier transforms which tell you "what frequencies exist,"
               wavelets tell you "what frequencies exist <em>and when</em>."</p>

            <h3>How to Read Scalograms</h3>
            <table>
                <tr><th>Axis</th><th>Meaning</th><th>Interpretation</th></tr>
                <tr><td>X-axis</td><td>Time</td><td>Position in the signal</td></tr>
                <tr><td>Y-axis</td><td>Scale (≈ period)</td><td>Small = high freq, Large = low freq</td></tr>
                <tr><td>Color</td><td>Power</td><td>Bright = strong signal at that scale/time</td></tr>
            </table>

            <div class="key-insight">
                <strong>Key Insight:</strong> Horizontal bands = persistent frequency. Vertical streaks = transient events.
                The "ski slope" widening occurs because longer periods require more time context to detect.
            </div>
        </section>

        <section class="section highlight">
            <h2>1. Basic Waveforms</h2>
            <p>Let's see how fundamental waveforms appear in the wavelet domain. Each shape has a distinct "fingerprint."</p>

            <div class="chart">
                <img src="data:image/png;base64,{plots['basic_waves']}" alt="Basic Waveforms">
            </div>

            <h3>Observations</h3>
            <ul>
                <li><strong>Sine wave:</strong> Single horizontal band - pure tone has one frequency</li>
                <li><strong>Square wave:</strong> Main band + fainter bands above (odd harmonics: 3f, 5f, 7f...)</li>
                <li><strong>Sawtooth:</strong> Main band + many harmonics (all integer multiples)</li>
                <li><strong>Triangle:</strong> Similar to square but harmonics decay faster (1/n² vs 1/n)</li>
            </ul>
        </section>

        <section class="section success">
            <h2>2. Frequency Variations</h2>
            <p>What happens when frequency changes? The chirp signal is especially revealing.</p>

            <div class="chart">
                <img src="data:image/png;base64,{plots['frequency_signals']}" alt="Frequency Variations">
            </div>

            <h3>Observations</h3>
            <ul>
                <li><strong>Slow sine:</strong> Horizontal band at large scale (low frequency = long period)</li>
                <li><strong>Fast sine:</strong> Horizontal band at small scale (high frequency = short period)</li>
                <li><strong>Chirp:</strong> Diagonal sweep! Frequency increases over time, and the wavelet tracks it perfectly</li>
            </ul>

            <div class="key-insight">
                <strong>Key Insight:</strong> The chirp demonstrates wavelets' superpower: tracking frequency changes over time.
                Fourier would just show a smeared blob; wavelets show the exact trajectory.
            </div>
        </section>

        <section class="section warning">
            <h2>3. Transient Events</h2>
            <p>How do sudden events appear? This explains the "flames" we see in real data.</p>

            <div class="chart">
                <img src="data:image/png;base64,{plots['transients']}" alt="Transient Events">
            </div>

            <h3>Observations</h3>
            <ul>
                <li><strong>Spike:</strong> Vertical line across ALL scales - a delta contains all frequencies</li>
                <li><strong>Step:</strong> Vertical event at transition, strongest at large scales (low freq = DC shift)</li>
                <li><strong>Gaussian pulse:</strong> Localized blob - the "ideal" wavelet response, compact in both domains</li>
                <li><strong>Damped oscillation:</strong> Bright region that fades horizontally as amplitude decays</li>
            </ul>

            <div class="key-insight">
                <strong>Key Insight:</strong> The spike creates a vertical "flame" because an instantaneous event
                is equivalent to superimposing ALL frequencies at that moment. This is why Wikipedia article spikes
                create those ski-slope patterns.
            </div>
        </section>

        <section class="section">
            <h2>4. Signal Operations</h2>
            <p>What happens in the wavelet domain when we combine signals through addition, multiplication, or division?</p>

            <div class="chart">
                <img src="data:image/png;base64,{plots['operations']}" alt="Signal Operations">
            </div>

            <h3>Addition: slow + fast</h3>
            <p>The wavelet spectrum shows <strong>both frequency bands</strong> - addition preserves the individual components.
               You see two distinct horizontal bands at their respective scales.</p>
            <div class="formula">Addition → Superposition in wavelet domain</div>

            <h3>Multiplication: slow × fast (Amplitude Modulation)</h3>
            <p>This is <strong>amplitude modulation</strong> (AM radio works this way). The fast signal's amplitude
               varies with the slow signal. In wavelet domain, you see the fast frequency band, but it now has
               <strong>periodic intensity variations</strong> matching the slow signal's period.</p>
            <div class="formula">Multiplication → Modulated intensity patterns</div>

            <h3>Division: fast ÷ |slow|</h3>
            <p>Division creates <strong>inverse amplitude modulation</strong> - the fast signal gets louder when
               the slow signal is small. The wavelet shows intensity peaks where the divisor approaches zero,
               creating periodic "hot spots."</p>
            <div class="formula">Division → Inverse modulation (poles at zeros)</div>

            <div class="key-insight">
                <strong>Key Insight:</strong> Addition keeps frequencies separate. Multiplication/division create
                <em>interactions</em> between frequencies - you see the fast frequency modulated by the slow one.
                This is how you'd detect if Wikipedia traffic patterns are being modulated by external cycles.
            </div>
        </section>

        <section class="section highlight">
            <h2>5. The Uncertainty Principle</h2>
            <p>Why do the "flames" widen at larger scales? This is fundamental physics.</p>

            <h3>Time-Frequency Tradeoff</h3>
            <p>You cannot simultaneously know the exact time AND exact frequency of an event.
               This is analogous to Heisenberg's uncertainty principle in quantum mechanics:</p>

            <div class="formula">Δt × Δf ≥ constant</div>

            <table>
                <tr><th>Scale</th><th>Time Resolution</th><th>Frequency Resolution</th></tr>
                <tr><td>Small (high freq)</td><td>Excellent</td><td>Poor</td></tr>
                <tr><td>Large (low freq)</td><td>Poor</td><td>Excellent</td></tr>
            </table>

            <p>To detect a 365-day cycle, you need ~365 days of context. A spike that happens on one day
               gets "smeared" across that entire year when viewed at the yearly scale - hence the widening.</p>

            <div class="key-insight">
                <strong>Key Insight:</strong> The ski slopes aren't a bug - they're a feature! They show the
                fundamental limit of time-frequency analysis. The slope angle is determined by the wavelet's
                time-frequency resolution (Morlet wavelet ≈ 45°).
            </div>
        </section>

        <section class="section success">
            <h2>Summary: Pattern Recognition Guide</h2>

            <table>
                <tr><th>Pattern</th><th>Wavelet Signature</th><th>Real-World Example</th></tr>
                <tr><td>Pure periodic</td><td>Horizontal band</td><td>Weekly TV show interest</td></tr>
                <tr><td>Changing frequency</td><td>Diagonal sweep</td><td>Accelerating trend</td></tr>
                <tr><td>Sudden spike</td><td>Vertical flame</td><td>Breaking news event</td></tr>
                <tr><td>Step change</td><td>Vertical + low-freq glow</td><td>Article goes viral permanently</td></tr>
                <tr><td>Damped oscillation</td><td>Fading horizontal</td><td>News cycle decay</td></tr>
                <tr><td>Multiple periodicities</td><td>Multiple bands</td><td>Weekly + yearly patterns</td></tr>
                <tr><td>Modulated signal</td><td>Band with varying intensity</td><td>Seasonal amplitude changes</td></tr>
            </table>
        </section>

        <footer>
            Wavelet Transform Tutorial &bull; Built with PyWavelets (Morlet wavelet)
        </footer>
    </div>
</body>
</html>'''

    return html


def main():
    print("Generating wavelet tutorial...")
    setup_plot_style()

    plots = {}

    # 1. Basic waveforms
    print("  Generating basic waveforms...")
    t, signals = generate_signals()
    plots['basic_waves'] = plot_comparison_grid(t, signals, 'Basic Waveforms: Time vs Wavelet Domain')

    # 2. Frequency variations
    print("  Generating frequency variations...")
    t, signals = generate_frequency_signals()
    plots['frequency_signals'] = plot_comparison_grid(t, signals, 'Frequency Variations')

    # 3. Transients
    print("  Generating transient signals...")
    t, signals = generate_transient_signals()
    plots['transients'] = plot_comparison_grid(t, signals, 'Transient Events')

    # 4. Operations
    print("  Generating signal operations...")
    t, signals = generate_compound_signals()
    plots['operations'] = plot_operation_demo(t, signals, 'Signal Operations: Addition, Multiplication, Division')

    # Generate HTML
    print("  Generating HTML report...")
    html = generate_html(plots)

    # Save
    REPORTS_DIR.mkdir(exist_ok=True)
    output_file = REPORTS_DIR / 'wavelet_tutorial.html'

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_file}")
    print(f"Open in browser: file://{output_file.absolute()}")


if __name__ == '__main__':
    main()
