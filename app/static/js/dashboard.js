/**
 * Dashboard interactivity and API orchestration for ReviewGuard-AI.
 */

let currentBatchResults = [];

// Sample reviews data cache
const SAMPLE_PRESETS = [
    {
        text: "BEST PRODUCT EVERRRR!!!! Absolutely blown away by the quality!! Changed my life forever!! MUST BUY 100% recommended to everyone!! A++++++ service!!!",
        rating: "5.0",
        category: "General"
    },
    {
        text: "I've been using this wireless earbuds for about 6 months. Sound stage is solid and ANC noise cancellation works as expected. The companion mobile app could be improved, but overall decent value for $79.",
        rating: "4.0",
        category: "Electronics"
    },
    {
        text: "DO NOT BUY!!! TOTAL SCAM!! Broke in 5 seconds!! Garbage trash worthless junk!! Worst company on planet earth!! RUN AWAY SCAM SCAM SCAM!!",
        rating: "1.0",
        category: "Electronics"
    },
    {
        text: "Completely broken on arrival, total disaster and useless waste of money. Worst thing ever.",
        rating: "5.0",
        category: "General"
    }
];

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize text counter if inspector form present
    const reviewTextarea = document.getElementById('review-text');
    if (reviewTextarea) {
        reviewTextarea.addEventListener('input', updateCharWordCount);
    }

    // 2. Initialize drag-and-drop for batch file upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('batch-file-input');
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-brand-500', 'bg-gray-900/80');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-brand-500', 'bg-gray-900/80');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-brand-500', 'bg-gray-900/80');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleBatchFileSelected(fileInput.files[0]);
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleBatchFileSelected(fileInput.files[0]);
            }
        });
    }

    // 3. Initialize Benchmark Leaderboard if on benchmark page
    if (document.getElementById('benchmark-table-body')) {
        loadBenchmarkLeaderboard();
    }
});

function updateCharWordCount() {
    const text = document.getElementById('review-text').value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    const counter = document.getElementById('char-word-counter');
    if (counter) {
        counter.textContent = `${words} words | ${chars} chars`;
    }
}

function loadSample(index) {
    if (index < 0 || index >= SAMPLE_PRESETS.length) return;
    const sample = SAMPLE_PRESETS[index];
    const textarea = document.getElementById('review-text');
    const ratingSelect = document.getElementById('review-rating');
    const catSelect = document.getElementById('review-category');

    if (textarea) textarea.value = sample.text;
    if (ratingSelect) ratingSelect.value = sample.rating;
    if (catSelect) catSelect.value = sample.category;

    updateCharWordCount();
    inspectReview();
}

/**
 * Executes review inspection and explainability analysis.
 */
async function inspectReview() {
    const textInput = document.getElementById('review-text');
    const ratingInput = document.getElementById('review-rating');
    const btn = document.getElementById('inspect-btn');
    const resultsContainer = document.getElementById('results-container');

    const text = textInput ? textInput.value.trim() : '';
    if (!text) {
        alert('Please enter review text to inspect.');
        return;
    }

    const ratingVal = ratingInput && ratingInput.value ? parseFloat(ratingInput.value) : null;

    // Show loading state
    const originalBtnHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Analyzing Transformers & Saliency...</span>`;

    try {
        const response = await fetch('/api/v1/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, rating: ratingVal })
        });

        const json = await response.json();
        if (!response.ok || json.status !== 'success') {
            throw new Error(json.error || 'Analysis failed.');
        }

        const data = json.data;
        renderInspectionResults(data);
        resultsContainer.classList.remove('hidden');
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        alert('Error analyzing review: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalBtnHtml;
    }
}

/**
 * Updates DOM with full explainability analysis results.
 */
function renderInspectionResults(data) {
    const pred = data.prediction;
    const tokens = data.tokens || [];
    const diag = data.diagnostics || {};
    const sent = data.sentiment || {};
    const sty = data.stylometrics || {};
    const radar = data.radar_data || {};
    const topWords = data.top_words || {};

    // 1. KPI: Risk Level & Icon
    const riskEl = document.getElementById('kpi-risk-level');
    const iconBox = document.getElementById('kpi-icon-box');
    const icon = document.getElementById('kpi-icon');
    const consensusEl = document.getElementById('kpi-consensus');

    if (riskEl) riskEl.textContent = pred.risk_level.toUpperCase();
    if (consensusEl) consensusEl.textContent = pred.consensus;

    if (pred.risk_badge === 'danger') {
        iconBox.className = 'w-12 h-12 rounded-xl flex items-center justify-center text-xl bg-red-500/10 text-red-400 border border-red-500/20';
        icon.className = 'fa-solid fa-triangle-exclamation';
        riskEl.className = 'text-lg font-extrabold text-red-400';
    } else if (pred.risk_badge === 'warning') {
        iconBox.className = 'w-12 h-12 rounded-xl flex items-center justify-center text-xl bg-amber-500/10 text-amber-400 border border-amber-500/20';
        icon.className = 'fa-solid fa-circle-exclamation';
        riskEl.className = 'text-lg font-extrabold text-amber-400';
    } else {
        iconBox.className = 'w-12 h-12 rounded-xl flex items-center justify-center text-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        icon.className = 'fa-solid fa-shield-check';
        riskEl.className = 'text-lg font-extrabold text-emerald-400';
    }

    // 2. KPI: Fake Probability Bar
    const fakePercent = (pred.fake_probability * 100).toFixed(1);
    const fakePercentEl = document.getElementById('kpi-fake-percent');
    const fakeBarEl = document.getElementById('kpi-fake-bar');
    if (fakePercentEl) fakePercentEl.textContent = fakePercent + '%';
    if (fakeBarEl) {
        fakeBarEl.style.width = fakePercent + '%';
        if (pred.fake_probability >= 0.70) fakeBarEl.className = 'h-2.5 rounded-full bg-red-500';
        else if (pred.fake_probability >= 0.45) fakeBarEl.className = 'h-2.5 rounded-full bg-amber-500';
        else fakeBarEl.className = 'h-2.5 rounded-full bg-emerald-500';
    }

    // 3. KPI: Sentiment & Dissonance
    const sentValEl = document.getElementById('kpi-sentiment-val');
    const dissBadge = document.getElementById('kpi-dissonance-badge');
    const labelCap = (sent.sentiment_label || sent.label || 'neutral').toUpperCase();
    if (sentValEl) {
        const sign = sent.polarity > 0 ? '+' : '';
        sentValEl.textContent = `${sign}${sent.polarity.toFixed(2)} (${labelCap})`;
    }
    if (dissBadge) {
        if (sent.is_dissonant) {
            dissBadge.className = 'text-xs px-2 py-0.5 rounded font-bold bg-red-500/20 text-red-300 border border-red-500/30';
            dissBadge.textContent = '⚠️ Severe Rating Dissonance!';
        } else {
            dissBadge.className = 'text-xs px-2 py-0.5 rounded font-medium bg-emerald-500/10 text-emerald-400';
            dissBadge.textContent = 'Aligned with Rating';
        }
    }

    // 4. KPI: Readability
    const fleschEl = document.getElementById('kpi-flesch-val');
    const readLevelEl = document.getElementById('kpi-readability-level');
    if (fleschEl) fleschEl.textContent = `${sty.flesch_reading_ease.toFixed(1)} / 100`;
    if (readLevelEl) {
        if (sty.flesch_reading_ease > 80) readLevelEl.textContent = 'Very Easy (Formulaic)';
        else if (sty.flesch_reading_ease > 60) readLevelEl.textContent = 'Standard Customer English';
        else readLevelEl.textContent = 'Complex / Technical Syntax';
    }

    // 5. Highlighted Tokens Box
    const textBox = document.getElementById('highlighted-text-box');
    if (textBox) {
        textBox.innerHTML = data.highlighted_html || '<p class="text-gray-400">Attribution unavailable.</p>';
    }

    // 6. Top Words Drivers Pills
    const fakeList = document.getElementById('top-fake-words-list');
    const genList = document.getElementById('top-genuine-words-list');
    if (fakeList) {
        fakeList.innerHTML = '';
        const fWords = topWords.top_fake_triggers || [];
        if (fWords.length === 0) {
            fakeList.innerHTML = '<span class="text-gray-500 italic">No significant fake triggers</span>';
        } else {
            fWords.forEach(w => {
                fakeList.innerHTML += `<span class="px-2 py-1 rounded bg-red-500/20 text-red-300 border border-red-500/30 font-mono">"${w.token}" (+${w.saliency.toFixed(2)})</span>`;
            });
        }
    }
    if (genList) {
        genList.innerHTML = '';
        const gWords = topWords.top_genuine_signals || [];
        if (gWords.length === 0) {
            genList.innerHTML = '<span class="text-gray-500 italic">No strong genuine drivers</span>';
        } else {
            gWords.forEach(w => {
                genList.innerHTML += `<span class="px-2 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">"${w.token}" (${w.saliency.toFixed(2)})</span>`;
            });
        }
    }

    // 7. Diagnostics Reasoner Cards
    const summaryEl = document.getElementById('diagnostic-summary-text');
    if (summaryEl) summaryEl.textContent = diag.summary || '';

    const redFlagsContainer = document.getElementById('red-flags-container');
    const redCountEl = document.getElementById('red-flags-count');
    if (redFlagsContainer) {
        redFlagsContainer.innerHTML = '';
        const rf = diag.red_flags || [];
        if (redCountEl) redCountEl.textContent = rf.length;
        if (rf.length === 0) {
            redFlagsContainer.innerHTML = '<p class="text-xs text-gray-500 italic">Zero anomaly flags detected. Clean profile.</p>';
        } else {
            rf.forEach(flag => {
                redFlagsContainer.innerHTML += `
                    <div class="p-2.5 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-200">
                        <strong class="text-red-400">[${flag.category}]</strong>: ${flag.message}
                    </div>
                `;
            });
        }
    }

    const authContainer = document.getElementById('authentic-signals-container');
    const authCountEl = document.getElementById('authentic-signals-count');
    if (authContainer) {
        authContainer.innerHTML = '';
        const as = diag.authentic_signals || [];
        if (authCountEl) authCountEl.textContent = as.length;
        if (as.length === 0) {
            authContainer.innerHTML = '<p class="text-xs text-gray-500 italic">No authentic markers recorded.</p>';
        } else {
            as.forEach(sig => {
                authContainer.innerHTML += `
                    <div class="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-200">
                        <strong class="text-emerald-400">[${sig.category}]</strong>: ${sig.message}
                    </div>
                `;
            });
        }
    }

    // 8. Model Voting Breakdown
    const modelVotesBox = document.getElementById('model-votes-container');
    if (modelVotesBox && pred.model_scores) {
        modelVotesBox.innerHTML = '';
        Object.entries(pred.model_scores).forEach(([mName, score]) => {
            const pct = (score * 100).toFixed(1);
            const isFakeVote = score >= 0.50;
            const badgeColor = isFakeVote ? 'text-red-400 bg-red-500/10' : 'text-emerald-400 bg-emerald-500/10';
            const cleanName = mName.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            modelVotesBox.innerHTML += `
                <div class="flex items-center justify-between py-1 border-b border-gray-800/60">
                    <span class="text-gray-300 font-medium">${cleanName}</span>
                    <span class="px-2 py-0.5 rounded text-[11px] font-mono font-bold ${badgeColor}">
                        ${pct}% ${isFakeVote ? 'FAKE' : 'GENUINE'}
                    </span>
                </div>
            `;
        });
    }

    // 9. Render Stylometric Radar Chart
    if (radar.labels && radar.review_values) {
        renderRadarChart(radar);
    }
}

/**
 * Batch Auditor Handlers
 */
function handleBatchFileSelected(file) {
    const nameDisplay = document.getElementById('file-name-display');
    if (nameDisplay && file) {
        nameDisplay.textContent = `Selected file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        nameDisplay.classList.remove('hidden');
    }
}

async function submitBatchFile() {
    const fileInput = document.getElementById('batch-file-input');
    if (!fileInput || !fileInput.files.length) {
        alert('Please choose or drag-and-drop a CSV file first.');
        return;
    }

    const btn = document.getElementById('upload-audit-btn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Auditing Batch Dataset...</span>`;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch('/api/v1/batch', {
            method: 'POST',
            body: formData,
        });
        const json = await res.json();
        if (!res.ok || json.status !== 'success') {
            throw new Error(json.error || 'Batch audit failed.');
        }
        renderBatchResults(json.summary, json.results);
    } catch (e) {
        alert('Batch audit error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function submitBatchPastedText() {
    const textarea = document.getElementById('batch-paste-textarea');
    const text = textarea ? textarea.value.trim() : '';
    if (!text) {
        alert('Please paste reviews to audit.');
        return;
    }

    const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const reviews = lines.map((line, idx) => ({ id: idx + 1, text: line }));

    try {
        const res = await fetch('/api/v1/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reviews: reviews })
        });
        const json = await res.json();
        if (!res.ok || json.status !== 'success') {
            throw new Error(json.error || 'Batch audit failed.');
        }
        renderBatchResults(json.summary, json.results);
    } catch (e) {
        alert('Batch audit error: ' + e.message);
    }
}

async function loadSampleCsvBatch() {
    // Curated sample list from data/sample_reviews.csv
    const samples = [
        { id: 1, text: "BEST PRODUCT EVERRRR!!!! Absolutely blown away by the quality!! Changed my life forever!! MUST BUY 100% recommended to everyone!! A++++++ service!!!", rating: 5.0 },
        { id: 2, text: "I've been using this wireless earbuds for about 6 months. The sound stage is solid and ANC noise cancellation works as expected. However, the companion mobile app could be improved. Overall decent value for $79.", rating: 4.0 },
        { id: 3, text: "DO NOT BUY!!! TOTAL SCAM!! Broke in 5 seconds!! Garbage trash worthless junk!! Worst company on planet earth!! RUN AWAY SCAM SCAM SCAM!!", rating: 1.0 },
        { id: 4, text: "Battery life on this smartwatch lasts roughly 18 hours of continuous use. Build quality feels sturdy with a matte finish. Minor gripe: the power cable is only 3 feet long. Would rate 4/5.", rating: 4.0 },
        { id: 5, text: "Completely broken on arrival, total disaster and useless waste of money. Worst thing ever.", rating: 5.0 },
        { id: 6, text: "We purchased this air fryer for daily meal prep. Cuts prep time in half and dishwasher cleaning is straightforward. The lid latch requires a firm press to seal properly.", rating: 4.0 },
        { id: 7, text: "DO YOURSELF A FAVOR AND BUY THIS RIGHT NOW!!! Hands down the greatest purchase I have ever made in my entire life!! Miracle product, five stars all the way!!", rating: 5.0 },
        { id: 8, text: "Sole separated from the toe box after 4 weeks of light walking on paved sidewalks. Material feels thin compared to older models from this brand.", rating: 1.0 },
        { id: 9, text: "Free sample provided for my honest review! Wow unbelievable quality, definitely worth every single penny, click the link to get your discount code!!", rating: 5.0 },
        { id: 10, text: "Fabric is 95% cotton and 5% elastane. Fits true to size around the chest (I am 6'0, 180 lbs, wearing Large). Shrunk slightly after first wash in warm water.", rating: 4.0 },
        { id: 11, text: "Nice product. Very nice quality. Very very good. Recommend this product to all buyers.", rating: 5.0 },
        { id: 12, text: "Thoroughly researched 340-page guide with clear diagrams and real-world case studies in chapters 4 and 7. The index at the back makes quick reference simple.", rating: 5.0 }
    ];

    try {
        const res = await fetch('/api/v1/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reviews: samples })
        });
        const json = await res.json();
        if (json.status === 'success') {
            renderBatchResults(json.summary, json.results);
        }
    } catch (e) {
        alert('Failed loading sample batch: ' + e.message);
    }
}

function renderBatchResults(summary, results) {
    currentBatchResults = results;
    const container = document.getElementById('batch-results-container');
    if (!container) return;

    // Summary Cards
    document.getElementById('batch-kpi-total').textContent = summary.total_reviews;
    document.getElementById('batch-kpi-flagged').textContent = summary.flagged_deceptive;
    document.getElementById('batch-kpi-percent').textContent = `${summary.deceptive_percentage}% deceptive rate`;
    document.getElementById('batch-kpi-authentic').textContent = summary.authentic_count;
    document.getElementById('batch-kpi-avg-score').textContent = summary.average_fake_probability.toFixed(3);

    // Table Records
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';
    results.forEach((r, idx) => {
        const probPct = (r.fake_probability * 100).toFixed(1);
        let badgeClass = 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        if (r.fake_probability >= 0.70) badgeClass = 'bg-red-500/10 text-red-400 border border-red-500/20 font-bold';
        else if (r.fake_probability >= 0.45) badgeClass = 'bg-amber-500/10 text-amber-400 border border-amber-500/20';

        const dissFlag = r.is_dissonant ? `<span class="px-1.5 py-0.5 rounded bg-red-500/20 text-red-300 font-bold">YES</span>` : `<span class="text-gray-500">No</span>`;
        const ratingStr = r.rating ? `${r.rating}★` : `<span class="text-gray-500">-</span>`;

        tbody.innerHTML += `
            <tr class="hover:bg-gray-800/40 transition-colors">
                <td class="py-3 px-4 font-mono font-bold text-gray-400">#${r.id || idx + 1}</td>
                <td class="py-3 px-4 max-w-md truncate" title="${r.text}">${r.text}</td>
                <td class="py-3 px-4 text-center font-medium">${ratingStr}</td>
                <td class="py-3 px-4 font-mono font-bold">${probPct}%</td>
                <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[11px] ${badgeClass}">${r.risk_level}</span></td>
                <td class="py-3 px-4 text-center font-mono font-semibold ${r.red_flags_count > 0 ? 'text-red-400' : 'text-gray-500'}">${r.red_flags_count}</td>
                <td class="py-3 px-4 text-center">${dissFlag}</td>
            </tr>
        `;
    });

    container.classList.remove('hidden');
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Render Charts
    renderBatchDistributionChart(results);
    renderBatchPieChart(results);
}

function exportBatchResultsCsv() {
    if (!currentBatchResults.length) {
        alert('No audit results to export.');
        return;
    }

    const headers = ["ID", "Review_Text", "Rating", "Fake_Probability", "Risk_Level", "Red_Flags_Count", "Is_Dissonant"];
    const rows = currentBatchResults.map(r => [
        r.id,
        `"${(r.text || '').replace(/"/g, '""')}"`,
        r.rating || '',
        (r.fake_probability || 0).toFixed(4),
        r.risk_level || '',
        r.red_flags_count || 0,
        r.is_dissonant ? 'True' : 'False'
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `reviewguard_audit_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Benchmark Leaderboard Loader
 */
async function loadBenchmarkLeaderboard() {
    try {
        const res = await fetch('/api/v1/models');
        const json = await res.json();
        if (json.status !== 'success') return;

        const metrics = json.benchmark_metrics || {};
        const tbody = document.getElementById('benchmark-table-body');
        if (tbody) {
            tbody.innerHTML = '';
            const modelsSorted = Object.entries(metrics).sort((a, b) => b[1].f1_score - a[1].f1_score);

            const chartLabels = [];
            const f1Data = [];
            const aucData = [];

            modelsSorted.forEach(([name, m], idx) => {
                const cleanName = name.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                const isWinner = idx === 0;
                const badge = isWinner ? '<span class="ml-2 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px] font-bold uppercase">Rank #1</span>' : '';
                const typeName = name.includes('transformer') ? 'Deep Learning Transformer' : (name.includes('boost') ? 'Gradient Boosted Trees' : 'Linear / Probabilistic ML');

                tbody.innerHTML += `
                    <tr class="hover:bg-gray-800/40 transition-colors">
                        <td class="py-3 px-6 font-bold text-white flex items-center">${cleanName} ${badge}</td>
                        <td class="py-3 px-6 font-mono">${(m.accuracy * 100).toFixed(2)}%</td>
                        <td class="py-3 px-6 font-mono">${(m.precision * 100).toFixed(2)}%</td>
                        <td class="py-3 px-6 font-mono">${(m.recall * 100).toFixed(2)}%</td>
                        <td class="py-3 px-6 font-mono font-bold text-emerald-400">${(m.f1_score * 100).toFixed(2)}%</td>
                        <td class="py-3 px-6 font-mono text-brand-400">${(m.roc_auc * 100).toFixed(2)}%</td>
                        <td class="py-3 px-6 text-xs text-gray-400">${typeName}</td>
                    </tr>
                `;

                chartLabels.push(cleanName);
                f1Data.push((m.f1_score * 100).toFixed(1));
                aucData.push((m.roc_auc * 100).toFixed(1));
            });

            // Benchmark comparison chart
            const ctx = document.getElementById('benchmark-chart');
            if (ctx) {
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: chartLabels,
                        datasets: [
                            {
                                label: 'F1-Score (%)',
                                data: f1Data,
                                backgroundColor: 'rgba(16, 185, 129, 0.8)',
                                borderRadius: 4,
                            },
                            {
                                label: 'ROC-AUC (%)',
                                data: aucData,
                                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                                borderRadius: 4,
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false,
                                min: 60,
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
                            legend: {
                                labels: { color: '#d1d5db' }
                            }
                        }
                    }
                });
            }
        }

        // HF status pill
        const hfPill = document.getElementById('hf-status-pill');
        if (hfPill && json.huggingface_status) {
            if (json.huggingface_status.hf_available) {
                hfPill.className = 'text-xs px-2.5 py-0.5 rounded-full font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
                hfPill.textContent = 'PyTorch & HF Ready';
            } else {
                hfPill.className = 'text-xs px-2.5 py-0.5 rounded-full font-semibold bg-brand-500/20 text-brand-300 border border-brand-500/30';
                hfPill.textContent = 'Native NumPy/Keras Transformer Active';
            }
        }
    } catch (e) {
        console.error('Error loading benchmark leaderboard:', e);
    }
}

/**
 * XAI Lab Threshold Simulator
 */
function updateThreshold(val) {
    const display = document.getElementById('threshold-val-display');
    const note = document.getElementById('threshold-policy-note');
    if (display) display.textContent = parseFloat(val).toFixed(2);

    const v = parseFloat(val);
    if (!note) return;

    if (v <= 0.25) {
        note.innerHTML = `<strong>Aggressive Moderation Policy (${v.toFixed(2)}):</strong> Maximizes fraud recall at the expense of precision. Flags even subtle marketing hyperbole as suspicious. Suitable for high-fraud marketplaces.`;
    } else if (v >= 0.75) {
        note.innerHTML = `<strong>Conservative Audit Policy (${v.toFixed(2)}):</strong> Maximizes precision to protect genuine sellers from false accusations. Flags only reviews with extreme anomalous markers and confirmed model consensus.`;
    } else {
        note.innerHTML = `<strong>Standard Balanced Policy (${v.toFixed(2)}):</strong> Optimal F1-score balance. Reviews with probability >= ${v.toFixed(2)} are categorized as Suspicious or Deceptive.`;
    }
}

/**
 * Interactive API Console
 */
function updateApiPresetPayload(endpoint) {
    const bodyArea = document.getElementById('api-request-body');
    const curlSnippet = document.getElementById('curl-snippet');
    if (!bodyArea) return;

    if (endpoint === '/api/v1/explain') {
        bodyArea.value = JSON.stringify({
            text: "BEST PRODUCT EVERRRR!!!! Changed my life, must buy 100% recommended!!",
            rating: 5.0
        }, null, 2);
        if (curlSnippet) curlSnippet.textContent = `curl -X POST http://127.0.0.1:5000/api/v1/explain -H "Content-Type: application/json" -d '{"text":"BEST PRODUCT EVERRRR!!","rating":5.0}'`;
    } else if (endpoint === '/api/v1/predict') {
        bodyArea.value = JSON.stringify({
            text: "I've used these headphones for 3 months. Solid battery, good sound.",
            rating: 4.0
        }, null, 2);
        if (curlSnippet) curlSnippet.textContent = `curl -X POST http://127.0.0.1:5000/api/v1/predict -H "Content-Type: application/json" -d '{"text":"I used these headphones...","rating":4.0}'`;
    } else if (endpoint === '/api/v1/batch') {
        bodyArea.value = JSON.stringify({
            reviews: [
                { id: 1, text: "BEST PRODUCT EVER!! Buy now!", rating: 5.0 },
                { id: 2, text: "Solid build, fast delivery.", rating: 4.0 }
            ]
        }, null, 2);
        if (curlSnippet) curlSnippet.textContent = `curl -X POST http://127.0.0.1:5000/api/v1/batch -H "Content-Type: application/json" -d '{"reviews":[{"id":1,"text":"BEST EVER"}]}'`;
    } else if (endpoint === '/api/v1/models') {
        bodyArea.value = "{}";
        if (curlSnippet) curlSnippet.textContent = `curl -X GET http://127.0.0.1:5000/api/v1/models`;
    } else if (endpoint === '/api/v1/samples') {
        bodyArea.value = "{}";
        if (curlSnippet) curlSnippet.textContent = `curl -X GET http://127.0.0.1:5000/api/v1/samples`;
    }
}

async function sendApiConsoleRequest() {
    const endpoint = document.getElementById('api-endpoint-select').value;
    const bodyText = document.getElementById('api-request-body').value;
    const outPre = document.getElementById('api-response-output');
    const statusBadge = document.getElementById('api-status-badge');
    const sendBtn = document.getElementById('api-send-btn');

    const originalBtn = sendBtn.innerHTML;
    sendBtn.disabled = true;
    sendBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Sending...`;

    try {
        const isPost = endpoint.includes('explain') || endpoint.includes('predict') || endpoint.includes('batch');
        const opts = {
            method: isPost ? 'POST' : 'GET',
            headers: isPost ? { 'Content-Type': 'application/json' } : {}
        };
        if (isPost && bodyText.trim()) {
            opts.body = bodyText;
        }

        const res = await fetch(endpoint, opts);
        const json = await res.json();

        statusBadge.textContent = `Status: ${res.status} ${res.statusText}`;
        statusBadge.className = res.ok ? 'text-xs px-2 py-0.5 rounded font-mono bg-emerald-500/20 text-emerald-300' : 'text-xs px-2 py-0.5 rounded font-mono bg-red-500/20 text-red-300';
        outPre.textContent = JSON.stringify(json, null, 2);

    } catch (e) {
        statusBadge.textContent = 'Status: Error';
        statusBadge.className = 'text-xs px-2 py-0.5 rounded font-mono bg-red-500/20 text-red-300';
        outPre.textContent = '// Request Error: ' + e.message;
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = originalBtn;
    }
}
