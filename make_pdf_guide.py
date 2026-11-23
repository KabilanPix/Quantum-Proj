import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib import colors

PDF_OUTPUT_PATH = Path(__file__).resolve().parent / "Quantum_Digital_Twin_Project_Guide.pdf"

def generate_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284C7'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Title Banner
    story.append(Paragraph("🏭 Quantum Digital Twin for Smart Factories", title_style))
    story.append(Paragraph("Complete Technical Architecture, System Modules & Explanation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # Executive Summary Box
    summary_html = (
        "<b>Project Mission:</b> Modern smart factories struggle because sensor monitoring, predictive maintenance, "
        "and job scheduling operate in silos. This project builds a unified <b>Cyber-Physical Digital Twin</b>: "
        "IoT sensors stream real-time telemetry (temperature, vibration, power) to a REST API and SQLite database; "
        "machine learning models predict breakdown risk and detect anomalies; and a <b>Quantum QAOA optimizer</b> "
        "dynamically steers production jobs away from failing machines while automatically scheduling maintenance."
    )
    summary_table = Table([[Paragraph(summary_html, callout_style)]], colWidths=[532])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F9FF')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#BAE6FD')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Section 1: End-to-End Pipeline
    story.append(Paragraph("1. The 4-Stage End-to-End System Architecture", h1_style))
    pipeline_data = [
        [Paragraph("<b>Stage</b>", body_style), Paragraph("<b>Component</b>", body_style), Paragraph("<b>How It Works & What It Does</b>", body_style)],
        [
            Paragraph("<b>1. Ingestion</b>", body_style),
            Paragraph("Physics Simulator & ESP32 Virtual Firmware", body_style),
            Paragraph("Generates temperature, vibration, and power telemetry across realistic failure scenarios (bearing wear, overheating, power surge) with sensor noise, drift, and dropouts.", body_style)
        ],
        [
            Paragraph("<b>2. Backend</b>", body_style),
            Paragraph("Flask REST API + SQLite WAL", body_style),
            Paragraph("Exposes /sensor-data, /latest, /events, /risk, /schedule endpoints. Persists telemetry in high-concurrency WAL mode with automatic schema migration.", body_style)
        ],
        [
            Paragraph("<b>3. Intelligence</b>", body_style),
            Paragraph("ML Risk, Anomaly & Quantum Scheduler", body_style),
            Paragraph("Random Forest & Gradient Boosting predict failure probabilities. Isolation Forest detects anomalies. QAOA QUBO optimizes job assignments and assigns maintenance.", body_style)
        ],
        [
            Paragraph("<b>4. Frontend</b>", body_style),
            Paragraph("Streamlit Dashboard", body_style),
            Paragraph("Interactive executive dashboard with live charts, failure risk trends, energy forecasts, real dataset validation tabs, and what-if simulation panels.", body_style)
        ]
    ]
    t_pipe = Table(pipeline_data, colWidths=[80, 130, 322])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_pipe)
    story.append(Spacer(1, 10))

    # Section 2: Machine Learning & Predictive Maintenance
    story.append(Paragraph("2. Machine Learning & Predictive Maintenance", h1_style))
    story.append(Paragraph(
        "The digital twin deploys a dual-tier intelligence layer: supervised predictive classification and unsupervised anomaly detection.",
        body_style
    ))
    story.append(Paragraph("• <b>Feature Engineering (10 Features):</b> Rolling 30-step windows calculate mean, standard deviation, max, and slopes (linear regression drift) for temperature, vibration, and power.", bullet_style))
    story.append(Paragraph("• <b>Synthetic Failure Modes:</b> The model is trained across distinct degradation mechanics: <i>Bearing Wear</i> (exponential vibration growth), <i>Overheating</i> (steep temperature rise), and <i>Power Surges</i> (electrical spikes).", bullet_style))
    story.append(Paragraph("• <b>Real Industrial Benchmark (UCI AI4I 2020):</b> Validated against 10,000 real industrial milling machine records. Because genuine breakdowns are rare (~3.4%), we emphasize <b>Recall (80.9%)</b>, <b>F1-Score (0.85)</b>, and <b>PR-AUC (0.83)</b> over raw accuracy.", bullet_style))
    story.append(Paragraph("• <b>Unsupervised Anomaly Detection:</b> An <i>Isolation Forest</i> fits normal operating clusters per machine to flag abnormal sensor drift before threshold violations occur.", bullet_style))
    story.append(Paragraph("• <b>Energy Forecaster:</b> Autoregressive lag-feature Gradient Boosting predicts future 30-minute machine power draw, feeding energy costs into the scheduler.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 3: Quantum Scheduling & QUBO Formulation
    story.append(Paragraph("3. Quantum Optimization (QAOA) & QUBO Formulation", h1_style))
    story.append(Paragraph(
        "Job-to-machine allocation is formulated as a <b>Quadratic Unconstrained Binary Optimization (QUBO)</b> problem:",
        body_style
    ))
    
    qubo_box = (
        "<b>Objective Hamiltonian:</b><br/>"
        "<b>Cost = &Sigma;<sub>(j, m)</sub> [ Energy<sub>j</sub> + &alpha; &middot; Risk<sub>m</sub> &middot; Duration<sub>j</sub> ] &middot; x<sub>j,m</sub> "
        "+ &beta; &middot; &Sigma;<sub>m</sub> ( Load<sub>m</sub> - TargetLoad )<sup>2</sup> + &gamma; &middot; MaintenancePenalty</b><br/><br/>"
        "<b>Subject to:</b> &Sigma;<sub>m</sub> x<sub>j,m</sub> = 1 (Every job assigned to exactly one machine).<br/>"
        "<b>Where:</b> x<sub>j,m</sub> &isin; {0, 1} are binary decision variables mapped directly to quantum qubits."
    )
    t_qubo = Table([[Paragraph(qubo_box, body_style)]], colWidths=[532])
    t_qubo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_qubo)
    story.append(Spacer(1, 6))

    story.append(Paragraph("• <b>How QAOA Solves It:</b> Constraints are converted into penalty terms. QAOA applies alternating problem and mixer Hamiltonians parameterized by angles (&gamma;, &beta;) and optimizes expectation values using COBYLA on a quantum simulator.", bullet_style))
    story.append(Paragraph("• <b>Graceful Classical Fallback:</b> In case of simulator timeout or missing quantum libraries, the system falls back seamlessly to an exact classical brute-force solver, ensuring 100% production uptime.", bullet_style))
    story.append(Paragraph("• <b>Proactive Maintenance Slot:</b> When machine risk crosses 70%, the scheduler dynamically inserts a dedicated <code>MAINT_M</code> task to repair the machine and prevents heavy production jobs from overloading it.", bullet_style))

    story.append(PageBreak())

    # Section 4: Virtual & Physical Hardware Layer
    story.append(Paragraph("4. ESP32 Virtual Hardware & Physical Firmware Layer", h1_style))
    story.append(Paragraph(
        "To guarantee that software simulated in this phase can transfer directly to real factory hardware with zero rework, production-ready embedded firmware was authored.",
        body_style
    ))
    story.append(Paragraph("• <b>Embedded Sensors:</b> <code>DHT22</code> (ambient temperature on GPIO 15), <code>MPU6050</code> (3-axis I2C accelerometer on GPIO 21/22 computing g-force vibration magnitude), and a <code>Potentiometer</code> (analog ADC on GPIO 34 scaling 0-6 kW motor power).", bullet_style))
    story.append(Paragraph("• <b>Wokwi Cloud Simulation:</b> Schematics (<code>diagram.json</code>) and C++ code (<code>sketch.ino</code>) wired and ready to run inside the Wokwi browser simulator.", bullet_style))
    story.append(Paragraph("• <b>Serial Bridge (scripts/serial_bridge.py):</b> Reads JSON lines from physical USB COM ports or simulated mock lines and POSTs them to the API. Every reading carries a <code>source</code> tag ('simulator', 'esp32-sim', or 'esp32-real').", bullet_style))
    story.append(Spacer(1, 10))

    # Section 5: Real-World Benchmark Results
    story.append(Paragraph("5. Empirical Validation & Benchmarking Results", h1_style))
    bench_data = [
        [Paragraph("<b>Model / Formulation</b>", body_style), Paragraph("<b>Dataset / Target</b>", body_style), Paragraph("<b>Recall</b>", body_style), Paragraph("<b>F1 Score</b>", body_style), Paragraph("<b>ROC-AUC</b>", body_style), Paragraph("<b>Nature</b>", body_style)],
        [
            Paragraph("Random Forest", body_style),
            Paragraph("Synthetic Multi-Mode Physics", body_style),
            Paragraph("83.2%", body_style),
            Paragraph("0.895", body_style),
            Paragraph("0.976", body_style),
            Paragraph("Simulated", body_style)
        ],
        [
            Paragraph("Gradient Boosting (Best)", body_style),
            Paragraph("UCI AI4I 2020 (Real 10k)", body_style),
            Paragraph("80.9%", body_style),
            Paragraph("0.853", body_style),
            Paragraph("0.977", body_style),
            Paragraph("Real Industrial", body_style)
        ],
        [
            Paragraph("Random Forest", body_style),
            Paragraph("UCI AI4I 2020 (Real 10k)", body_style),
            Paragraph("83.8%", body_style),
            Paragraph("0.792", body_style),
            Paragraph("0.977", body_style),
            Paragraph("Real Industrial", body_style)
        ],
        [
            Paragraph("Logistic Regression", body_style),
            Paragraph("UCI AI4I 2020 (Real 10k)", body_style),
            Paragraph("86.8%", body_style),
            Paragraph("0.309", body_style),
            Paragraph("0.946", body_style),
            Paragraph("Real Industrial", body_style)
        ]
    ]
    t_bench = Table(bench_data, colWidths=[110, 140, 55, 60, 65, 102])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # Section 6: How to Run the Project
    story.append(Paragraph("6. Quick Start & Execution Commands", h1_style))
    commands_data = [
        [Paragraph("<b>Goal</b>", body_style), Paragraph("<b>Command Line Instruction</b>", body_style)],
        [Paragraph("<b>Start Everything (API, Simulator, Pipeline, Dashboard)</b>", body_style), Paragraph("<code>python scripts/run_all.py</code>", body_style)],
        [Paragraph("<b>Run Acceptance Pipeline Checks</b>", body_style), Paragraph("<code>python scripts/test_pipeline.py</code>", body_style)],
        [Paragraph("<b>Run Pytest Unit Test Suite (8/8 tests)</b>", body_style), Paragraph("<code>pytest tests/ -v</code>", body_style)],
        [Paragraph("<b>Retrain Real Industrial ML Models</b>", body_style), Paragraph("<code>python ml/real_data/train_real.py</code>", body_style)],
        [Paragraph("<b>Run QAOA vs Classical Benchmark</b>", body_style), Paragraph("<code>python quantum/benchmark.py --quick</code>", body_style)],
        [Paragraph("<b>Run ESP32 Mock Telemetry Bridge</b>", body_style), Paragraph("<code>python scripts/serial_bridge.py --mock</code>", body_style)]
    ]
    t_cmd = Table(commands_data, colWidths=[200, 332])
    t_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cmd)
    story.append(Spacer(1, 10))

    # Section 7: Key Talking Points for Mentors / Evaluators
    story.append(Paragraph("7. Key Talking Points (How to Explain to Anyone)", h1_style))
    
    tp_text = (
        "<b>1. Why Quantum Scheduling?</b> Standard factory schedulers assign jobs statically based on duration. "
        "Our system turns machine degradation risk into mathematical penalty weights inside a QUBO matrix, so a quantum "
        "or classical optimizer automatically steers high-value jobs away from failing machines.<br/>"
        "<b>2. Why Prioritize Recall?</b> In real predictive maintenance (like UCI AI4I 2020), 96.6% of data is healthy. "
        "Predicting healthy 100% of the time yields 96.6% accuracy but misses every breakdown. Recall and F1 prove we catch true failures.<br/>"
        "<b>3. Is Hardware Real or Simulated?</b> Telemetry is generated via realistic differential physics and virtual ESP32 firmware, "
        "while ML is validated on real industrial datasets. The firmware is 100% genuine C++ ready to flash to physical microcontrollers."
    )
    t_tp = Table([[Paragraph(tp_text, body_style)]], colWidths=[532])
    t_tp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#FCD34D')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_tp)

    doc.build(story)
    print(f"[PDF] Successfully created {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    generate_pdf()
