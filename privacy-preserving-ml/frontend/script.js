// Privacy-Preserving ML Dashboard JavaScript

// Global variables
let currentData = {};
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    loadInitialData();
    setupEventListeners();
    initializeCharts();
});

// Tab functionality
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Refresh charts when tab is activated
            setTimeout(() => refreshChartsInActiveTab(targetTab), 100);
        });
    });
}

// Load initial data
async function loadInitialData() {
    showLoading();
    
    try {
        // Simulate loading data from backend
        currentData = await loadExperimentData();
        updateDashboard();
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load experiment data');
    } finally {
        hideLoading();
    }
}

// Simulate data loading (replace with actual API calls)
async function loadExperimentData() {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
        baseline: {
            'logistic_regression': { accuracy: 0.9420, roc_auc: 0.9820, f1_score: 0.8136 },
            'random_forest': { accuracy: 0.9485, roc_auc: 0.9875, f1_score: 0.8245 },
            'gradient_boosting': { accuracy: 0.9505, roc_auc: 0.9886, f1_score: 0.8312 },
            'svm': { accuracy: 0.9395, roc_auc: 0.9798, f1_score: 0.8089 }
        },
        differential_privacy: {
            '0.1': { accuracy: 0.7520, roc_auc: 0.7845, f1_score: 0.6234 },
            '0.5': { accuracy: 0.8234, roc_auc: 0.8456, f1_score: 0.7123 },
            '1.0': { accuracy: 0.8756, roc_auc: 0.8923, f1_score: 0.7654 },
            '2.0': { accuracy: 0.9123, roc_auc: 0.9234, f1_score: 0.7987 },
            '5.0': { accuracy: 0.9345, roc_auc: 0.9456, f1_score: 0.8123 },
            '10.0': { accuracy: 0.9423, roc_auc: 0.9567, f1_score: 0.8234 }
        },
        federated_learning: {
            rounds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            accuracy: [0.7234, 0.7856, 0.8123, 0.8345, 0.8567, 0.8723, 0.8834, 0.8923, 0.9012, 0.9087],
            roc_auc: [0.7456, 0.8012, 0.8234, 0.8456, 0.8678, 0.8823, 0.8934, 0.9023, 0.9112, 0.9187]
        },
        experiments: [
            { name: 'DP Experiment ε=1.0', date: '2024-01-15', status: 'completed', auc: 0.8923 },
            { name: 'Federated Learning 5 clients', date: '2024-01-14', status: 'completed', auc: 0.9187 },
            { name: 'Baseline Comparison', date: '2024-01-13', status: 'completed', auc: 0.9886 }
        ]
    };
}

// Update dashboard with loaded data
function updateDashboard() {
    updateStats();
    updateRecentExperiments();
    updateModelPerformance();
    updatePrivacyInsights();
}

// Update statistics cards
function updateStats() {
    const bestBaseline = Math.max(...Object.values(currentData.baseline).map(m => m.roc_auc));
    document.getElementById('best-auc').textContent = bestBaseline.toFixed(4);
    
    const modelCount = Object.keys(currentData.baseline).length;
    document.getElementById('models-trained').textContent = modelCount;
    
    const privacyMethods = 2; // DP and FL
    document.getElementById('privacy-methods').textContent = privacyMethods;
}

// Update recent experiments
function updateRecentExperiments() {
    const experimentList = document.getElementById('experimentList');
    experimentList.innerHTML = '';
    
    currentData.experiments.forEach(exp => {
        const expElement = document.createElement('div');
        expElement.className = 'experiment-item';
        expElement.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>${exp.name}</h4>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">${exp.date}</p>
                </div>
                <div style="text-align: right;">
                    <span class="status-badge ${exp.status}">${exp.status}</span>
                    <p style="font-weight: 600; color: var(--primary-color);">AUC: ${exp.auc.toFixed(4)}</p>
                </div>
            </div>
        `;
        experimentList.appendChild(expElement);
    });
}

// Initialize charts
function initializeCharts() {
    initializeModelComparisonChart();
    initializePrivacyBudgetChart();
    initializePrivacyUtilityChart();
    initializeUtilityDropChart();
    initializePrivacyEfficiencyChart();
}

// Model comparison chart
function initializeModelComparisonChart() {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    
    const models = Object.keys(currentData.baseline);
    const aucs = models.map(model => currentData.baseline[model].roc_auc);
    const accuracies = models.map(model => currentData.baseline[model].accuracy);
    
    charts.modelComparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => m.replace('_', ' ').toUpperCase()),
            datasets: [{
                label: 'ROC-AUC',
                data: aucs,
                backgroundColor: 'rgba(37, 99, 235, 0.8)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 2
            }, {
                label: 'Accuracy',
                data: accuracies,
                backgroundColor: 'rgba(16, 185, 129, 0.8)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.7,
                    max: 1.0
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// Privacy budget distribution chart
function initializePrivacyBudgetChart() {
    const epsilons = Object.keys(currentData.differential_privacy).map(parseFloat);
    const aucs = Object.values(currentData.differential_privacy).map(d => d.roc_auc);
    
    const trace = {
        x: epsilons,
        y: aucs,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'ROC-AUC',
        line: {
            color: 'rgb(37, 99, 235)',
            width: 3
        },
        marker: {
            color: 'rgb(37, 99, 235)',
            size: 8
        }
    };
    
    const layout = {
        title: '',
        xaxis: {
            title: 'Privacy Budget (ε)',
            type: 'log'
        },
        yaxis: {
            title: 'ROC-AUC',
            range: [0.7, 1.0]
        },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('privacyBudgetChart', [trace], layout, {responsive: true});
}

// Privacy-utility tradeoff chart
function initializePrivacyUtilityChart() {
    const epsilons = Object.keys(currentData.differential_privacy).map(parseFloat);
    const aucs = Object.values(currentData.differential_privacy).map(d => d.roc_auc);
    const accuracies = Object.values(currentData.differential_privacy).map(d => d.accuracy);
    
    const trace1 = {
        x: epsilons,
        y: aucs,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'ROC-AUC',
        line: { color: 'rgb(37, 99, 235)', width: 3 },
        marker: { color: 'rgb(37, 99, 235)', size: 8 }
    };
    
    const trace2 = {
        x: epsilons,
        y: accuracies,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Accuracy',
        line: { color: 'rgb(16, 185, 129)', width: 3 },
        marker: { color: 'rgb(16, 185, 129)', size: 8 }
    };
    
    // Add baseline reference line
    const baselineAUC = Math.max(...Object.values(currentData.baseline).map(m => m.roc_auc));
    const trace3 = {
        x: [Math.min(...epsilons), Math.max(...epsilons)],
        y: [baselineAUC, baselineAUC],
        mode: 'lines',
        type: 'scatter',
        name: 'Baseline (No Privacy)',
        line: { color: 'rgb(239, 68, 68)', width: 2, dash: 'dash' }
    };
    
    const layout = {
        title: '',
        xaxis: {
            title: 'Privacy Budget (ε)',
            type: 'log'
        },
        yaxis: {
            title: 'Performance',
            range: [0.7, 1.0]
        },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        legend: {
            x: 0.7,
            y: 0.3
        }
    };
    
    Plotly.newPlot('privacyUtilityChart', [trace1, trace2, trace3], layout, {responsive: true});
}

// Utility drop chart
function initializeUtilityDropChart() {
    const ctx = document.getElementById('utilityDropChart').getContext('2d');
    
    const baselineAUC = Math.max(...Object.values(currentData.baseline).map(m => m.roc_auc));
    const epsilons = Object.keys(currentData.differential_privacy);
    const utilityDrops = epsilons.map(eps => 
        baselineAUC - currentData.differential_privacy[eps].roc_auc
    );
    
    charts.utilityDrop = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: epsilons.map(eps => `ε=${eps}`),
            datasets: [{
                label: 'Utility Drop (AUC)',
                data: utilityDrops,
                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                borderColor: 'rgba(239, 68, 68, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Privacy efficiency chart
function initializePrivacyEfficiencyChart() {
    const ctx = document.getElementById('privacyEfficiencyChart').getContext('2d');
    
    const epsilons = Object.keys(currentData.differential_privacy).map(parseFloat);
    const aucs = Object.values(currentData.differential_privacy).map(d => d.roc_auc);
    const efficiency = aucs.map((auc, i) => auc * epsilons[i]);
    
    charts.privacyEfficiency = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epsilons.map(eps => `ε=${eps}`),
            datasets: [{
                label: 'Privacy Efficiency (AUC × ε)',
                data: efficiency,
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Update privacy insights
function updatePrivacyInsights() {
    const insightsContainer = document.getElementById('privacyInsights');
    
    const baselineAUC = Math.max(...Object.values(currentData.baseline).map(m => m.roc_auc));
    const bestDPAUC = Math.max(...Object.values(currentData.differential_privacy).map(d => d.roc_auc));
    const utilityDrop = baselineAUC - bestDPAUC;
    const relativeDrop = (utilityDrop / baselineAUC) * 100;
    
    const insights = [
        {
            title: 'Privacy Cost',
            content: `Best private model achieves ${bestDPAUC.toFixed(4)} AUC with ${utilityDrop.toFixed(4)} drop (${relativeDrop.toFixed(1)}%) from baseline.`
        },
        {
            title: 'Optimal Privacy Budget',
            content: 'ε = 1.0-5.0 provides good privacy-utility balance for most applications.'
        },
        {
            title: 'Strong Privacy',
            content: 'ε < 1.0 provides strong privacy guarantees but with significant utility loss.'
        },
        {
            title: 'Recommendation',
            content: 'Consider federated learning combined with local DP for maximum privacy protection.'
        }
    ];
    
    insightsContainer.innerHTML = insights.map(insight => `
        <div class="insight-card">
            <h4>${insight.title}</h4>
            <p>${insight.content}</p>
        </div>
    `).join('');
}

// Event listeners
function setupEventListeners() {
    // Privacy parameter controls
    const epsilonSlider = document.getElementById('epsilonSlider');
    const epsilonValue = document.getElementById('epsilonValue');
    
    epsilonSlider.addEventListener('input', (e) => {
        epsilonValue.textContent = e.target.value;
    });
    
    document.getElementById('updatePrivacyBtn').addEventListener('click', updatePrivacyAnalysis);
    
    // Model selection
    document.querySelectorAll('.model-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            updateModelDetails(e.target.dataset.model);
        });
    });
    
    // Experiment controls
    document.getElementById('runExperimentBtn').addEventListener('click', runNewExperiment);
    
    // Settings
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
    
    // Range sliders
    setupRangeSliders();
}

// Setup range sliders with value display
function setupRangeSliders() {
    const sliders = [
        { slider: 'trainSplit', display: 'trainSplitValue', format: (v) => `${(v*100).toFixed(0)}%` },
        { slider: 'animationSpeed', display: 'animationSpeedValue', format: (v) => `${v}ms` }
    ];
    
    sliders.forEach(({slider, display, format}) => {
        const sliderEl = document.getElementById(slider);
        const displayEl = document.getElementById(display);
        
        if (sliderEl && displayEl) {
            sliderEl.addEventListener('input', (e) => {
                displayEl.textContent = format(e.target.value);
            });
        }
    });
}

// Update privacy analysis
async function updatePrivacyAnalysis() {
    showLoading();
    
    const epsilon = parseFloat(document.getElementById('epsilonSlider').value);
    const delta = parseFloat(document.getElementById('deltaInput').value);
    
    try {
        // Simulate running new analysis
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Update charts with new parameters
        updatePrivacyInsights();
        showSuccess('Privacy analysis updated successfully!');
    } catch (error) {
        showError('Failed to update privacy analysis');
    } finally {
        hideLoading();
    }
}

// Update model details
function updateModelDetails(modelType) {
    const detailsContainer = document.getElementById('modelDetails');
    
    let content = '';
    switch(modelType) {
        case 'baseline':
            content = `
                <h3>Baseline Models (No Privacy)</h3>
                <p>Traditional machine learning models without privacy protection:</p>
                <ul>
                    <li><strong>Gradient Boosting:</strong> Best performance with ${currentData.baseline.gradient_boosting.roc_auc.toFixed(4)} AUC</li>
                    <li><strong>Random Forest:</strong> Robust ensemble method</li>
                    <li><strong>Logistic Regression:</strong> Interpretable linear model</li>
                    <li><strong>SVM:</strong> Support vector machine classifier</li>
                </ul>
            `;
            break;
        case 'dp':
            content = `
                <h3>Differential Privacy Models</h3>
                <p>Models with (ε,δ)-differential privacy guarantees:</p>
                <ul>
                    <li><strong>Privacy Budget:</strong> ε ∈ [0.1, 10.0], δ = 1e-5</li>
                    <li><strong>Mechanism:</strong> Output perturbation with Gaussian noise</li>
                    <li><strong>Best Performance:</strong> ε=10.0 achieves ${currentData.differential_privacy['10.0'].roc_auc.toFixed(4)} AUC</li>
                    <li><strong>Strong Privacy:</strong> ε=0.1 provides robust privacy with ${currentData.differential_privacy['0.1'].roc_auc.toFixed(4)} AUC</li>
                </ul>
            `;
            break;
        case 'fl':
            content = `
                <h3>Federated Learning</h3>
                <p>Distributed training without centralizing sensitive data:</p>
                <ul>
                    <li><strong>Clients:</strong> 5 simulated healthcare institutions</li>
                    <li><strong>Data Distribution:</strong> Non-IID across clients</li>
                    <li><strong>Final Performance:</strong> ${currentData.federated_learning.roc_auc[currentData.federated_learning.roc_auc.length-1].toFixed(4)} AUC</li>
                    <li><strong>Convergence:</strong> Stable after 10 rounds</li>
                </ul>
            `;
            break;
    }
    
    detailsContainer.innerHTML = content;
    updatePerformanceTable(modelType);
}

// Update performance table
function updatePerformanceTable(modelType) {
    const tableContainer = document.getElementById('performanceTable');
    
    let tableHTML = '<table class="performance-table"><thead><tr><th>Model/Config</th><th>Accuracy</th><th>ROC-AUC</th><th>F1-Score</th></tr></thead><tbody>';
    
    if (modelType === 'baseline') {
        Object.entries(currentData.baseline).forEach(([model, metrics]) => {
            tableHTML += `
                <tr>
                    <td>${model.replace('_', ' ').toUpperCase()}</td>
                    <td>${metrics.accuracy.toFixed(4)}</td>
                    <td>${metrics.roc_auc.toFixed(4)}</td>
                    <td>${metrics.f1_score.toFixed(4)}</td>
                </tr>
            `;
        });
    } else if (modelType === 'dp') {
        Object.entries(currentData.differential_privacy).forEach(([epsilon, metrics]) => {
            tableHTML += `
                <tr>
                    <td>ε = ${epsilon}</td>
                    <td>${metrics.accuracy.toFixed(4)}</td>
                    <td>${metrics.roc_auc.toFixed(4)}</td>
                    <td>${metrics.f1_score.toFixed(4)}</td>
                </tr>
            `;
        });
    }
    
    tableHTML += '</tbody></table>';
    tableContainer.innerHTML = tableHTML;
}

// Run new experiment
async function runNewExperiment() {
    const progressContainer = document.getElementById('experimentProgress');
    const resultsContainer = document.getElementById('experimentResults');
    const progressText = document.getElementById('progressText');
    
    progressContainer.style.display = 'block';
    resultsContainer.innerHTML = '';
    
    const experimentType = document.getElementById('experimentType').value;
    
    try {
        // Simulate experiment progress
        const steps = ['Initializing...', 'Loading data...', 'Training models...', 'Evaluating privacy...', 'Generating results...'];
        
        for (let i = 0; i < steps.length; i++) {
            progressText.textContent = steps[i];
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        // Show results
        resultsContainer.innerHTML = `
            <div class="experiment-result">
                <h4>Experiment Completed: ${experimentType.toUpperCase()}</h4>
                <p>✅ Successfully completed ${experimentType} experiment</p>
                <p>📊 Results added to dashboard</p>
                <p>⏱️ Execution time: ${steps.length} seconds</p>
            </div>
        `;
        
        showSuccess('Experiment completed successfully!');
    } catch (error) {
        showError('Experiment failed: ' + error.message);
    } finally {
        progressContainer.style.display = 'none';
    }
}

// Save settings
function saveSettings() {
    const settings = {
        datasetPath: document.getElementById('datasetPath').value,
        trainSplit: document.getElementById('trainSplit').value,
        defaultEpsilon: document.getElementById('defaultEpsilon').value,
        defaultDelta: document.getElementById('defaultDelta').value,
        budgetTracking: document.getElementById('budgetTracking').checked,
        chartTheme: document.getElementById('chartTheme').value,
        animationSpeed: document.getElementById('animationSpeed').value
    };
    
    localStorage.setItem('privacyMLSettings', JSON.stringify(settings));
    showSuccess('Settings saved successfully!');
}

// Export functions
function exportResults(format) {
    showLoading();
    
    setTimeout(() => {
        const data = JSON.stringify(currentData, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `privacy_ml_results.${format}`;
        a.click();
        URL.revokeObjectURL(url);
        
        hideLoading();
        showSuccess(`Results exported as ${format.toUpperCase()}!`);
    }, 1000);
}

// Utility functions
function refreshChartsInActiveTab(tabName) {
    Object.values(charts).forEach(chart => {
        if (chart && chart.resize) {
            chart.resize();
        }
    });
    
    // Refresh Plotly charts
    if (document.getElementById('privacyBudgetChart')) {
        Plotly.Plots.resize('privacyBudgetChart');
    }
    if (document.getElementById('privacyUtilityChart')) {
        Plotly.Plots.resize('privacyUtilityChart');
    }
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showSuccess(message) {
    // Simple success notification (you can enhance this)
    const notification = document.createElement('div');
    notification.className = 'notification success';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--success-color);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        z-index: 1001;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function showError(message) {
    const notification = document.createElement('div');
    notification.className = 'notification error';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--danger-color);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        z-index: 1001;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Initialize model performance on load
function updateModelPerformance() {
    updateModelDetails('baseline');
}

// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .performance-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    
    .performance-table th,
    .performance-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    
    .performance-table th {
        background: rgba(37, 99, 235, 0.1);
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .performance-table tr:hover {
        background: rgba(37, 99, 235, 0.05);
    }
    
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .status-badge.completed {
        background: rgba(16, 185, 129, 0.2);
        color: var(--success-color);
    }
    
    .experiment-result {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .experiment-result h4 {
        color: var(--success-color);
        margin-bottom: 10px;
    }
`;
document.head.appendChild(style);
