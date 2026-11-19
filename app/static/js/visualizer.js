/**
 * Visualizer & Chart.js helper module for ReviewGuard-AI.
 */

let radarChartInstance = null;
let batchDistChartInstance = null;
let batchPieChartInstance = null;
let benchmarkChartInstance = null;

/**
 * Initializes or updates the Stylometric Radar Chart.
 */
function renderRadarChart(radarData) {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: radarData.labels,
            datasets: [
                {
                    label: 'Analyzed Review',
                    data: radarData.review_values,
                    backgroundColor: 'rgba(99, 102, 241, 0.25)',
                    borderColor: '#6366f1',
                    pointBackgroundColor: '#818cf8',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#6366f1',
                    borderWidth: 2,
                },
                {
                    label: 'Authentic Benchmark',
                    data: radarData.benchmark_values,
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderColor: '#10b981',
                    pointBackgroundColor: '#34d399',
                    pointBorderColor: '#fff',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    pointLabels: {
                        color: '#9ca3af',
                        font: { size: 10, family: 'Inter' }
                    },
                    ticks: {
                        display: false,
                        backdropColor: 'transparent',
                        max: 100,
                        min: 0,
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#d1d5db',
                        font: { size: 11, family: 'Inter' },
                        boxWidth: 12,
                    }
                }
            }
        }
    });
}

/**
 * Renders batch scatter/bar distribution chart.
 */
function renderBatchDistributionChart(results) {
    const ctx = document.getElementById('batch-distribution-chart');
    if (!ctx) return;

    if (batchDistChartInstance) {
        batchDistChartInstance.destroy();
    }

    // Bucket by star rating or index
    const labels = results.map((r, i) => `#${i + 1}`);
    const probabilities = results.map(r => (r.fake_probability * 100).toFixed(1));
    const backgroundColors = results.map(r => {
        if (r.fake_probability >= 0.75) return 'rgba(239, 68, 68, 0.8)';
        if (r.fake_probability >= 0.50) return 'rgba(245, 158, 11, 0.8)';
        if (r.fake_probability >= 0.25) return 'rgba(59, 130, 246, 0.8)';
        return 'rgba(16, 185, 129, 0.8)';
    });

    batchDistChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Deceptive Probability (%)',
                data: probabilities,
                backgroundColor: backgroundColors,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#9ca3af', callback: v => v + '%' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

/**
 * Renders batch risk categorization donut chart.
 */
function renderBatchPieChart(results) {
    const ctx = document.getElementById('batch-pie-chart');
    if (!ctx) return;

    if (batchPieChartInstance) {
        batchPieChartInstance.destroy();
    }

    let authentic = 0;
    let likelyAuth = 0;
    let suspicious = 0;
    let highRisk = 0;

    results.forEach(r => {
        const p = r.fake_probability;
        if (p < 0.25) authentic++;
        else if (p < 0.50) likelyAuth++;
        else if (p < 0.75) suspicious++;
        else highRisk++;
    });

    batchPieChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Verified Authentic', 'Likely Authentic', 'Suspicious', 'High Risk / Fraud'],
            datasets: [{
                data: [authentic, likelyAuth, suspicious, highRisk],
                backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#d1d5db', font: { size: 10 } }
                }
            },
            cutout: '65%'
        }
    });
}
