document.addEventListener('DOMContentLoaded', () => {
    const activityLog = document.getElementById('activity-log');

    if (activityLog && typeof recentActivities !== 'undefined') {
        recentActivities.forEach(activity => {
            const item = document.createElement('div');
            item.className = 'bg-gray-900 p-4 rounded-lg border-l-4 border-blue-500 hover:bg-gray-700 transition cursor-pointer';

            let icon = 'fa-tasks';
            let color = 'blue';

            if (activity.type === 'red_team') { icon = 'fa-user-secret'; color = 'red'; }
            if (activity.type === 'monitor') { icon = 'fa-eye'; color = 'green'; }
            if (activity.type === 'system') { icon = 'fa-server'; color = 'gray'; }

            item.style.borderColor = `var(--${color}-500)`; // Note: Tailwind classes handle colors, this is illustrative

            item.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex items-center">
                        <div class="p-2 bg-gray-800 rounded-lg mr-3">
                            <i class="fas ${icon} text-${color}-400"></i>
                        </div>
                        <div>
                            <h4 class="font-bold text-gray-200">${activity.title}</h4>
                            <p class="text-sm text-gray-400">${activity.details}</p>
                        </div>
                    </div>
                    <span class="text-xs text-gray-500 whitespace-nowrap">${activity.time}</span>
                </div>
            `;
            activityLog.appendChild(item);
        });
    }
});
