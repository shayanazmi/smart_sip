// ─── Shooting Stars Background System (Vanilla JS Port) ───
class ShootingStarSystem {
    constructor(svgId, configs) {
        this.svg = document.getElementById(svgId);
        this.configs = configs;
        this.activeStars = [];
        if (this.svg) this.init();
    }

    init() {
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        this.svg.appendChild(defs);

        this.configs.forEach((config, index) => {
            const gradId = `star-grad-${index}`;
            const linearGrad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
            linearGrad.setAttribute("id", gradId);
            linearGrad.setAttribute("x1", "0%");
            linearGrad.setAttribute("y1", "0%");
            linearGrad.setAttribute("x2", "100%");
            linearGrad.setAttribute("y2", "100%");
            
            const stop1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            stop1.setAttribute("offset", "0%");
            stop1.setAttribute("stop-color", config.trailColor);
            stop1.setAttribute("stop-opacity", "0");
            
            const stop2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            stop2.setAttribute("offset", "100%");
            stop2.setAttribute("stop-color", config.starColor);
            stop2.setAttribute("stop-opacity", "1");
            
            linearGrad.appendChild(stop1);
            linearGrad.appendChild(stop2);
            defs.appendChild(linearGrad);

            this.scheduleNextStar(config, gradId);
        });

        requestAnimationFrame(() => this.animate());
    }

    getRandomStartPoint() {
        const side = Math.floor(Math.random() * 4);
        const offsetX = Math.random() * window.innerWidth;
        const offsetY = Math.random() * window.innerHeight;

        switch (side) {
            case 0: return { x: offsetX, y: 0, angle: 45 };
            case 1: return { x: window.innerWidth, y: offsetY, angle: 135 };
            case 2: return { x: offsetX, y: window.innerHeight, angle: 225 };
            case 3: return { x: 0, y: offsetY, angle: 315 };
            default: return { x: 0, y: 0, angle: 45 };
        }
    }

    scheduleNextStar(config, gradId) {
        const delay = Math.random() * (config.maxDelay - config.minDelay) + config.minDelay;
        setTimeout(() => this.createStar(config, gradId), delay);
    }

    createStar(config, gradId) {
        if (!this.svg) return;
        const { x, y, angle } = this.getRandomStartPoint();
        const speed = Math.random() * (config.maxSpeed - config.minSpeed) + config.minSpeed;
        const starWidth = config.starWidth || 10;
        const starHeight = config.starHeight || 1;

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("fill", `url(#${gradId})`);
        
        const starObj = {
            el: rect, config, gradId,
            x, y, angle, speed, distance: 0, scale: 1, starWidth, starHeight
        };

        this.svg.appendChild(rect);
        this.activeStars.push(starObj);
    }

    animate() {
        for (let i = this.activeStars.length - 1; i >= 0; i--) {
            const star = this.activeStars[i];
            star.x += star.speed * Math.cos((star.angle * Math.PI) / 180);
            star.y += star.speed * Math.sin((star.angle * Math.PI) / 180);
            star.distance += star.speed;
            star.scale = 1 + star.distance / 100;

            if (star.x < -20 || star.x > window.innerWidth + 20 || 
                star.y < -20 || star.y > window.innerHeight + 20) {
                this.svg.removeChild(star.el);
                this.activeStars.splice(i, 1);
                this.scheduleNextStar(star.config, star.gradId);
                continue;
            }

            const w = star.starWidth * star.scale;
            const cx = star.x + w / 2;
            const cy = star.y + star.starHeight / 2;
            
            star.el.setAttribute("x", star.x);
            star.el.setAttribute("y", star.y);
            star.el.setAttribute("width", w);
            star.el.setAttribute("height", star.starHeight);
            star.el.setAttribute("transform", `rotate(${star.angle}, ${cx}, ${cy})`);
        }
        requestAnimationFrame(() => this.animate());
    }
}

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const elements = {
    // Nav
    statusDot: document.querySelector('.status-dot'),
    statusText: document.getElementById('market-status-text'),
    lastUpdated: document.getElementById('last-updated-time'),
    refreshBtn: document.getElementById('refresh-btn'),
    refreshIcon: document.getElementById('refresh-icon'),
    
    // Left Panel
    recCard: document.getElementById('rec-card'),
    regimeBadge: document.getElementById('regime-badge'),
    multiplierBadge: document.getElementById('multiplier-badge'),
    regimeMessage: document.getElementById('regime-message'),
    baseSipInput: document.getElementById('base-sip-input'),
    minTopupInput: document.getElementById('min-topup-input'),
    maxTopupInput: document.getElementById('max-topup-input'),
    topupAmount: document.getElementById('topup-amount'),
    totalAmount: document.getElementById('total-amount'),
    
    // Center & Right Panels
    gaugeScore: document.getElementById('gauge-score'),
    sentimentLabel: document.getElementById('sentiment-label'),
    aiExplanation: document.getElementById('ai-explanation'),
    metricTech: document.getElementById('metric-tech'),
    metricSent: document.getElementById('metric-sent'),
    marketCatalysts: document.getElementById('market-catalysts'),
    
    // Breakdown
    techScoreVal: document.getElementById('tech-score-val'),
    techProgress: document.getElementById('tech-progress'),
    sentScoreVal: document.getElementById('sent-score-val'),
    sentProgress: document.getElementById('sent-progress'),
    finalMultiVal: document.getElementById('final-multi-val'),
    
    // Deep Dive
    rsiVal: document.getElementById('rsi-val'),
    macdVal: document.getElementById('macd-val'),
    macdSigVal: document.getElementById('macd-sig-val'),
    bbUpperVal: document.getElementById('bb-upper-val'),
    bbLowerVal: document.getElementById('bb-lower-val'),
    articleCountVal: document.getElementById('article-count-val')
};

// State
let appState = {
    multiplier: 1.0,
    isDarkMode: false,
    charts: {} // Store chart instances to destroy/recreate
};

// ─── Event Listeners ───
elements.refreshBtn.addEventListener('click', () => initApp());
elements.baseSipInput.addEventListener('input', handleBaseSipChange);
elements.minTopupInput.addEventListener('input', handleBaseSipChange);
elements.maxTopupInput.addEventListener('input', handleBaseSipChange);

function initChartDefaults() {
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() || '#6b7280';
    const gridColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim() || '#e5e7eb';
    
    Chart.defaults.color = textColor;
    Chart.defaults.scale.grid.color = gridColor;
    Chart.defaults.font.family = "'Inter', sans-serif";
}

// ─── Initialization & Fetching ───
async function initApp() {
    setLoadingState(true);
    
    try {
        // Fetch recommendations (defaults to base_sip=5000)
        const baseSip = parseInt(elements.baseSipInput.value) || 5000;
        
        // Parallel fetching
        const [recRes, trendRes] = await Promise.all([
            fetch(`${API_BASE_URL}/get-recommendation?base_sip=${baseSip}`).catch(e => null),
            fetch(`${API_BASE_URL}/sentiment-trendline?days=30`).catch(e => null)
        ]);

        let recData, trendData;

        if (recRes && recRes.ok) {
            recData = await recRes.json();
            updateDashboard(recData);
        } else {
            // Fallback for demo if backend is offline
            console.warn("Backend offline, loading fallback data.");
            recData = getFallbackData();
            updateDashboard(recData);
        }

        if (trendRes && trendRes.ok) {
            trendData = await trendRes.json();
        } else {
             trendData = getFallbackTrendData();
        }
        
        // Render Charts
        initChartDefaults();
        renderCharts(recData, trendData);

    } catch (error) {
        console.error("Error initializing app:", error);
    } finally {
        setLoadingState(false);
        updateLastUpdatedTime();
    }
}

function setLoadingState(isLoading) {
    if (isLoading) {
        elements.refreshBtn.classList.add('spinning');
    } else {
        elements.refreshBtn.classList.remove('spinning');
    }
}

function updateLastUpdatedTime() {
    const now = new Date();
    const options = { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true };
    elements.lastUpdated.textContent = `Last updated: ${now.toLocaleString('en-IN', options)}`;
    
    // Fake Market hours (9:15 to 3:30 IST)
    const hours = now.getHours();
    const mins = now.getMinutes();
    const isWeekend = now.getDay() === 0 || now.getDay() === 6;
    
    let isOpen = false;
    if (!isWeekend) {
        if (hours > 9 || (hours === 9 && mins >= 15)) {
            if (hours < 15 || (hours === 15 && mins <= 30)) {
                isOpen = true;
            }
        }
    }
    
    if (isOpen) {
        elements.statusDot.className = 'status-dot open';
        elements.statusText.textContent = 'Market Open';
    } else {
        elements.statusDot.className = 'status-dot';
        elements.statusText.textContent = 'Market Closed';
    }
}

// ─── UI Updating Logic ───
function updateDashboard(data) {
    // 1. Store state for local recalculation
    appState.multiplier = data.final_multiplier;

    // 2. Regime and Styling
    const regime = data.regime.toUpperCase();
    elements.regimeBadge.textContent = regime;
    
    let colorVar;
    let message;
    let topupReason;
    if (regime === 'OVERSOLD') {
        colorVar = 'var(--regime-oversold)';
        message = 'Market is undervalued. A good time to increase your investment.';
        topupReason = 'AI multiplied by ' + data.final_multiplier.toFixed(2) + 'x to buy the market dip.';
    } else if (regime === 'OVERBOUGHT') {
        colorVar = 'var(--regime-overbought)';
        message = 'Market may be overvalued. Consider a smaller top-up this month.';
        topupReason = 'AI reduced multiplier to ' + data.final_multiplier.toFixed(2) + 'x to avoid market peaks.';
    } else {
        colorVar = 'var(--regime-neutral)';
        message = 'Market is fairly valued. Invest as per your regular plan.';
        topupReason = 'AI multiplier at ' + data.final_multiplier.toFixed(2) + 'x due to stable conditions.';
    }

    elements.regimeBadge.style.backgroundColor = colorVar;
    elements.regimeBadge.style.color = regime === 'NEUTRAL' ? '#000' : '#fff';
    
    elements.recCard.style.borderColor = colorVar;
    elements.topupAmount.style.color = colorVar;
    
    elements.regimeMessage.textContent = message;
    elements.regimeMessage.style.borderLeftColor = colorVar;

    const topupReasonEl = document.getElementById('topup-reason');
    if (topupReasonEl) topupReasonEl.textContent = topupReason;

    // 3. Multiplier & Amounts
    elements.multiplierBadge.textContent = `${data.final_multiplier.toFixed(2)}x MULTIPLIER`;
    
    // Update values based on current input as per requirements
    handleBaseSipChange();

    // 4. Center Panel (Gauge prep)
    const combinedScore = Math.round((data.technical_score * 0.6 + data.sentiment_score * 0.4) * 100);
    elements.gaugeScore.textContent = combinedScore;
    elements.sentimentLabel.textContent = data.sentiment_label;

    if (elements.metricTech) {
        elements.metricTech.textContent = data.technical_score > 0.6 ? 'Strong' : (data.technical_score < 0.4 ? 'Weak' : 'Neutral');
        elements.metricTech.className = data.technical_score > 0.6 ? 'positive' : (data.technical_score < 0.4 ? 'negative' : 'neutral');
    }
    if (elements.metricSent) {
        elements.metricSent.textContent = data.sentiment_label;
        elements.metricSent.className = data.sentiment_score > 0.6 ? 'positive' : (data.sentiment_score < 0.4 ? 'negative' : 'neutral');
    }

    if (elements.marketCatalysts && data.headlines) {
        const headlinesHtml = data.headlines.slice(0, 2).map(h => 
            `<div class="headline-item"><i data-lucide="newspaper"></i> <span>${h}</span></div>`
        ).join('');
        elements.marketCatalysts.innerHTML = headlinesHtml;
    }

    // 5. Right Panel (Explanation)
    elements.aiExplanation.innerHTML = data.explanation.replace(/\n/g, '<br/>');
    lucide.createIcons(); // Instantiates any icons embedded in the AI response

    // 6. Logic Breakdown
    elements.techScoreVal.textContent = data.technical_score.toFixed(2);
    elements.techProgress.style.width = `${data.technical_score * 100}%`;
    
    elements.sentScoreVal.textContent = data.sentiment_score.toFixed(2);
    elements.sentProgress.style.width = `${data.sentiment_score * 100}%`;
    
    elements.finalMultiVal.textContent = `${data.final_multiplier.toFixed(2)}x`;

    // 7. Indicators Deep Dive
    elements.rsiVal.textContent = data.indicators.rsi.toFixed(2);
    elements.macdVal.textContent = data.indicators.macd.toFixed(2);
    elements.macdSigVal.textContent = data.indicators.macd_signal.toFixed(2);
    elements.bbUpperVal.textContent = `₹${data.indicators.bb_upper.toFixed(0)}`;
    elements.bbLowerVal.textContent = `₹${data.indicators.bb_lower.toFixed(0)}`;
    elements.articleCountVal.textContent = `${data.article_count} Articles`;
}

// Recalculates top-up without API call
function handleBaseSipChange() {
    const baseSip = parseInt(elements.baseSipInput.value) || 0;
    const minTopup = parseInt(elements.minTopupInput.value) || 0;
    const maxTopup = parseInt(elements.maxTopupInput.value) || Infinity;
    
    // The AI multiplier applies to the BASE SIP to get the TOTAL targeted monthly investment
    const targetTotal = baseSip * appState.multiplier;
    
    // The top-up is the difference. We round to the nearest 100 to keep SIP step amounts uniform.
    let rawTopup = Math.round((targetTotal - baseSip) / 100) * 100;
    let finalTopup = rawTopup;
    
    // Constraint clamping
    if (finalTopup < minTopup) finalTopup = minTopup;
    if (finalTopup > maxTopup) finalTopup = maxTopup;
    
    const total = baseSip + finalTopup;

    const sign = finalTopup >= 0 ? '+' : '-';
    elements.topupAmount.textContent = `${sign}₹${Math.abs(finalTopup).toLocaleString()}`;
    elements.totalAmount.textContent = `₹${total.toLocaleString()}`;
    
    const rangeEl = document.getElementById('topup-range');
    if (rangeEl) {
        if (rawTopup !== finalTopup) {
            rangeEl.textContent = `(Capped! AI initially wanted ₹${rawTopup.toLocaleString()})`;
            rangeEl.style.color = 'var(--regime-overbought)'; // red/orange alert
        } else {
            rangeEl.textContent = `(Within your allowed limits)`;
            rangeEl.style.color = 'var(--text-muted)';
        }
    }
}


// ─── Charts Rendering ───
function renderCharts(data, trendData) {
    destroyCharts();

    // 1. Gauge Chart (Center Panel)
    const gaugeCtx = document.getElementById('gaugeChart');
    const combinedScore = (data.technical_score * 0.6 + data.sentiment_score * 0.4) * 100;
    
    appState.charts.gauge = new Chart(gaugeCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [combinedScore, 100 - combinedScore],
                backgroundColor: [
                    getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim() || '#2563eb',
                    getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim() || '#e5e7eb'
                ],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { tooltip: { enabled: false }, legend: { display: false } },
            animation: { animateRotate: true, animateScale: false }
        }
    });

    // Color definitions
    const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim();
    const purpleColor = '#8b5cf6';
    const pinkColor   = '#ec4899';
    const greenColor  = '#10b981';

    // 2. RSI Mock Chart (We only get the single latest value, so mock a trend)
    const rsiCtx = document.getElementById('rsiChart');
    const rsiTrend = [45, 48, 52, 50, 47, 55, Math.round(data.indicators.rsi)]; // Mock path leading to current
    
    appState.charts.rsi = new Chart(rsiCtx, {
        type: 'line',
        data: {
            labels: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'Ytd', 'Today'],
            datasets: [{
                label: 'RSI',
                data: rsiTrend,
                borderColor: purpleColor,
                backgroundColor: `${purpleColor}33`,
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 20, max: 80, grid: { color: 'rgba(0,0,0,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    // 3. MACD Mock Chart
    const macdCtx = document.getElementById('macdChart');
    appState.charts.macd = new Chart(macdCtx, {
        type: 'bar', // Histogram look
        data: {
            labels: ['D-4', 'D-3', 'D-2', 'Ytd', 'Today'],
            datasets: [{
                label: 'Histogram',
                data: [-2, 1, 3, 5, Math.round(data.indicators.macd - data.indicators.macd_signal)],
                backgroundColor: pinkColor,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });

    // 4. Bollinger Bands (Mock trend)
    const bbCtx = document.getElementById('bbChart');
    const clo = Math.round(data.indicators.close);
    const up  = Math.round(data.indicators.bb_upper);
    const lw  = Math.round(data.indicators.bb_lower);
    appState.charts.bb = new Chart(bbCtx, {
        type: 'line',
        data: {
            labels: ['D-4', 'D-3', 'D-2', 'Ytd', 'Today'],
            datasets: [
                {
                    label: 'Close',
                    data: [clo-400, clo-100, clo+200, clo+50, clo],
                    borderColor: accentColor,
                    borderWidth: 2,
                    tension: 0.2
                },
                {
                    label: 'Upper',
                    data: [up-300, up-150, up+100, up+50, up],
                    borderColor: 'rgba(100,116,139,0.3)',
                    borderWidth: 1,
                    borderDash: [5, 5]
                },
                {
                    label: 'Lower',
                    data: [lw-300, lw-150, lw+100, lw+50, lw],
                    borderColor: 'rgba(100,116,139,0.3)',
                    borderWidth: 1,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });

    // 5. Sentiment Trend Chart
    const sentCtx = document.getElementById('sentimentTrendChart');
    const sentLabels = trendData.map(d => d.date);
    const sentScores = trendData.map(d => d.sentiment_score * 100);

    appState.charts.sent = new Chart(sentCtx, {
        type: 'line',
        data: {
            labels: sentLabels,
            datasets: [{
                label: 'Sentiment Score',
                data: sentScores,
                borderColor: greenColor,
                backgroundColor: `${greenColor}22`,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { display: false }, y: { min: 0, max: 100 } }
        }
    });

    // 6. Backtesting Chart (Fixed vs SmartSIP) - 5 Years Dummy Data
    const backtestCtx = document.getElementById('backtestChart');
    
    // Generate 60 months of dummy data representing compounding growth
    const months = Array.from({length: 60}, (_, i) => `Month ${i+1}`);
    let fixedVal = 0, smartVal = 0;
    const fixedData = [], smartData = [];
    
    for(let i=0; i<60; i++) {
        fixedVal += 5000;
        // SmartSIP varies around 5000 based on regime
        const smartBase = 5000 * (0.8 + Math.random()*0.6); // Random multiplier mostly above 1 if good
        smartVal += smartBase;
        
        // Simulating 12% vs 16% CAGR
        fixedVal *= 1.01;
        smartVal *= 1.013;
        
        fixedData.push(Math.round(fixedVal));
        smartData.push(Math.round(smartVal));
    }

    appState.charts.backtest = new Chart(backtestCtx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'SmartSIP Strategy Portfolio Value',
                    data: smartData,
                    borderColor: accentColor,
                    backgroundColor: `${accentColor}11`,
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3
                },
                {
                    label: 'Fixed SIP Portfolio Value',
                    data: fixedData,
                    borderColor: '#9ca3af', // Gray
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            let val = context.parsed.y;
                            return ` ₹${val.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 12 } }, // Show roughly yearly
                y: {
                    ticks: { callback: v => `₹${(v/1000).toFixed(0)}k` }
                }
            }
        }
    });
}

function destroyCharts() {
    Object.keys(appState.charts).forEach(key => {
        if (appState.charts[key]) {
            appState.charts[key].destroy();
        }
    });
    appState.charts = {};
}


// ─── Dummy Data (Offline Mode Fallback) ───
function getFallbackData() {
    return {
        base_sip_amount: 5000,
        final_multiplier: 1.41,
        topup_amount: 7050,
        total_investment: 12050,
        technical_score: 0.85,
        sentiment_score: 0.72,
        regime: "Oversold",
        sentiment_label: "Neutral",
        predicted_close: 22105.4,
        current_close: 21950.2,
        ema_200: 21800.0,
        technical_weight: 0.6,
        sentiment_weight: 0.4,
        indicators: {
            rsi: 28.5,
            macd: -45.2,
            macd_signal: -20.1,
            bb_upper: 22500,
            bb_lower: 21800,
            bb_width: 0.03,
            ema_50: 22100,
            ema_200: 21800,
            cross_signal: -1,
            std_20: 350,
            close: 21950.2,
            log_return: -0.015
        },
        explanation: "<b>Market Insight:</b> The Nifty 50 is demonstrating deep oversold signals with an RSI under 30. Despite mixed news sentiment from recent earnings reports, our models detect a high probability bounce in the near term.<br><br><b>Recommended Action Plan:</b><ul class='action-list'><li><i data-lucide='trending-up'></i> <b>Accelerate Accumulation:</b> Capitalize on the current 4% dip in the index pricing by running a strong top-up multiplier.</li><li><i data-lucide='shield'></i> <b>Stay Disciplined:</b> Do not pause your base SIP during this psychologically volatile phase.</li><li><i data-lucide='briefcase'></i> <b>Sector Focus:</b> IT and Pharma are showing systemic relative strength against the wider market selloff.</li></ul>",
        headlines: ["Market dips on inflation fears", "Tech earnings surprise positively"],
        article_count: 24
    };
}

function getFallbackTrendData() {
    const data = [];
    let base = 0.5;
    for (let i = 29; i >= 0; i--) {
        const d = new Date();
        d.setDate(d.getDate() - i);
        base += (Math.random() - 0.45) * 0.1; 
        base = Math.max(0, Math.min(1, base));
        data.push({
            date: d.toISOString().split('T')[0],
            sentiment_score: base
        });
    }
    return data;
}

// ─── Boot ───
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    
    // Initialize the Shooting Stars Background exactly replicating the Demo props
    new ShootingStarSystem("shooting-stars-svg", [
        { starColor: "#9E00FF", trailColor: "#2EB9DF", minSpeed: 15, maxSpeed: 35, minDelay: 1000, maxDelay: 3000, starWidth: 10, starHeight: 1 },
        { starColor: "#FF0099", trailColor: "#FFB800", minSpeed: 10, maxSpeed: 25, minDelay: 2000, maxDelay: 4000, starWidth: 10, starHeight: 1 },
        { starColor: "#00FF9E", trailColor: "#00B8FF", minSpeed: 20, maxSpeed: 40, minDelay: 1500, maxDelay: 3500, starWidth: 10, starHeight: 1 }
    ]);
});
