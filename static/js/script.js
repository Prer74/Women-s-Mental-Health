const ctx = document.getElementById('sentimentChart').getContext('2d');
const sentimentChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        datasets: [{
            label: 'Sentiment Score',
            data: sentimentData,  // This will be set in your HTML
            backgroundColor: 'rgba(163, 70, 145, 0.2)',
            borderColor: 'rgba(163, 70, 145, 1)',
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                suggestedMax: 1,
                title: {
                    display: true,
                    text: 'Sentiment Score'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Day'
                }
            }
        }
    }
});
