document.addEventListener('DOMContentLoaded', () => {
    const savingsRange = document.getElementById('savingsRange');
    const spendingRange = document.getElementById('spendingRange');
    const debtRange = document.getElementById('debtRange');

    const avatar = document.getElementById('avatar-face');
    const statusPill = document.getElementById('health-status');
    const toast = document.getElementById('toast');

    function update() {
        const savings = parseInt(savingsRange.value);
        const spending = parseInt(spendingRange.value);
        const debt = parseInt(debtRange.value);

        // Update Labels
        document.getElementById('savingsVal').innerText = savings + '%';
        document.getElementById('spendingVal').innerText = '$' + spending;
        document.getElementById('debtVal').innerText = '$' + debt;

        // Logic
        let mood = 'neutral';
        let status = 'STABLE';

        if (debt > 5000) {
            mood = 'sick';
            status = 'CRITICAL';
        } else if (savings < 5) {
            mood = 'worried';
            status = 'AT RISK';
        } else if (savings > 25 && debt === 0) {
            mood = 'happy';
            status = 'OPTIMAL';
        }

        // Spending Intervention
        if (spending > 3000) {
            mood = 'angry';
            status = 'INTERVENING';
            showToast("I noticed you spent $500 on shoes. I have automatically cancelled your Netflix subscription to compensate. You're welcome.");
        } else {
            hideToast();
        }

        // Render Mood
        switch(mood) {
            case 'happy':
                avatar.innerText = '😎'; // Rich/Cool
                statusPill.style.background = '#d1fae5';
                statusPill.style.color = '#065f46';
                break;
            case 'neutral':
                avatar.innerText = '😐';
                statusPill.style.background = '#e2e8f0';
                statusPill.style.color = '#1e293b';
                break;
            case 'worried':
                avatar.innerText = '😰';
                statusPill.style.background = '#fef3c7';
                statusPill.style.color = '#92400e';
                break;
            case 'sick':
                avatar.innerText = '🤢';
                statusPill.style.background = '#fee2e2';
                statusPill.style.color = '#991b1b';
                break;
            case 'angry':
                avatar.innerText = '🤬';
                statusPill.style.background = '#7f1d1d';
                statusPill.style.color = '#fff';
                break;
        }
    }

    function showToast(msg) {
        toast.innerText = msg;
        toast.style.display = 'block';
    }

    function hideToast() {
        toast.style.display = 'none';
    }

    savingsRange.addEventListener('input', update);
    spendingRange.addEventListener('input', update);
    debtRange.addEventListener('input', update);

    update();
});
