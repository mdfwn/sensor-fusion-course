// Interactive features for Sensor Fusion Course

class CourseProgress {
    constructor() {
        this.storageKey = 'sensor_fusion_progress';
        this.progress = this.loadProgress();
        this.currentPage = this.getCurrentPage();
        this.init();
    }

    init() {
        this.createProgressTracker();
        this.initializeQuizzes();
        this.trackPageVisit();
        this.setupThemeToggle();
        this.setupMobileNavigation();
    }

    getCurrentPage() {
        return window.location.pathname.replace(/^\/|\/$/g, '').replace(/\//g, '_') || 'index';
    }

    loadProgress() {
        const saved = localStorage.getItem(this.storageKey);
        return saved ? JSON.parse(saved) : {
            visitedPages: [],
            completedQuizzes: [],
            currentCourse: null,
            lastVisited: null
        };
    }

    saveProgress() {
        localStorage.setItem(this.storageKey, JSON.stringify(this.progress));
    }

    trackPageVisit() {
        const currentPage = this.getCurrentPage();
        if (!this.progress.visitedPages.includes(currentPage)) {
            this.progress.visitedPages.push(currentPage);
        }
        this.progress.lastVisited = new Date().toISOString();
        this.saveProgress();
        this.updateProgressDisplay();
    }

    createProgressTracker() {
        // Create progress tracker element
        const tracker = document.createElement('div');
        tracker.className = 'progress-tracker';
        tracker.id = 'progress-tracker';
        
        const courses = [
            { name: 'Getting Started', pages: ['index', 'how_to_use', 'resources'] },
            { name: 'Lidar', pages: ['lidar_index', 'lidar_introduction', 'lidar_parsing_files', 'lidar_ransac', 'lidar_clustering', 'lidar_bounding_boxes', 'lidar_visualizer', 'lidar_summary'] },
            { name: 'Camera', pages: ['camera_index', 'camera_camera_models', 'camera_gradients', 'camera_features', 'camera_matching', 'camera_yolo', 'camera_tracking', 'camera_collision', 'camera_summary'] },
            { name: 'Radar', pages: ['radar_index', 'radar_principles', 'radar_fft', 'radar_cfar', 'radar_angle_estimation', 'radar_data_association', 'radar_multi_target', 'radar_summary'] },
            { name: 'Kalman Filters', pages: ['kalman_filters_index', 'kalman_filters_linear_kf', 'kalman_filters_extended_unscented', 'kalman_filters_noise_tuning', 'kalman_filters_transforms', 'kalman_filters_track_fusion', 'kalman_filters_imm', 'kalman_filters_misalignment', 'kalman_filters_summary'] },
            { name: 'Capstone', pages: ['capstone_index', 'capstone_brief', 'capstone_dataset', 'capstone_baseline', 'capstone_milestones', 'capstone_evaluation', 'capstone_submission'] }
        ];

        let trackerHTML = '<h3>📚 Course Progress</h3>';
        
        courses.forEach(course => {
            const completedPages = course.pages.filter(page => 
                this.progress.visitedPages.includes(page.replace(/\//g, '_'))
            );
            const progressPercent = Math.round((completedPages.length / course.pages.length) * 100);
            
            trackerHTML += `
                <div class="progress-item ${this.isCurrentCourse(course.pages) ? 'current' : ''}">
                    <span class="progress-icon">${progressPercent === 100 ? '✅' : '📖'}</span>
                    <span>${course.name} (${progressPercent}%)</span>
                </div>
            `;
        });

        // Add toggle button
        trackerHTML += `
            <button id="toggle-tracker" class="toggle-button" style="margin-top: 1rem; width: 100%;">
                Hide Progress
            </button>
        `;

        tracker.innerHTML = trackerHTML;
        document.body.appendChild(tracker);

        // Add toggle functionality
        document.getElementById('toggle-tracker').addEventListener('click', () => {
            tracker.classList.toggle('hidden');
            const button = document.getElementById('toggle-tracker');
            button.textContent = tracker.classList.contains('hidden') ? 'Show Progress' : 'Hide Progress';
        });
    }

    isCurrentCourse(coursePages) {
        return coursePages.some(page => page === this.currentPage);
    }

    updateProgressDisplay() {
        const tracker = document.getElementById('progress-tracker');
        if (tracker) {
            // Remove and recreate to update content
            tracker.remove();
            this.createProgressTracker();
        }
    }

    initializeQuizzes() {
        // Find all quiz containers and make them interactive
        document.querySelectorAll('.quiz-container').forEach(quiz => {
            this.setupQuiz(quiz);
        });
    }

    setupQuiz(quizContainer) {
        const quizId = quizContainer.dataset.quizId || 
                     `quiz_${Math.random().toString(36).substr(2, 9)}`;
        quizContainer.dataset.quizId = quizId;

        const options = quizContainer.querySelectorAll('.quiz-options li');
        const correctAnswer = quizContainer.dataset.correct;

        options.forEach(option => {
            option.addEventListener('click', () => {
                // Clear previous selections
                options.forEach(opt => opt.classList.remove('selected'));
                
                // Mark current selection
                option.classList.add('selected');
                
                // Check if answer is correct
                if (option.dataset.value === correctAnswer) {
                    this.markQuizCompleted(quizId);
                    this.showQuizFeedback(quizContainer, true, option.textContent);
                } else {
                    this.showQuizFeedback(quizContainer, false, option.textContent);
                }
            });
        });
    }

    markQuizCompleted(quizId) {
        if (!this.progress.completedQuizzes.includes(quizId)) {
            this.progress.completedQuizzes.push(quizId);
            this.saveProgress();
        }
    }

    showQuizFeedback(container, correct, answer) {
        // Remove existing feedback
        const existingFeedback = container.querySelector('.quiz-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }

        const feedback = document.createElement('div');
        feedback.className = 'quiz-feedback';
        feedback.style.cssText = `
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 6px;
            font-weight: 500;
            ${correct ? 
                'background: rgba(34, 197, 94, 0.1); color: rgb(34, 197, 94); border: 1px solid rgba(34, 197, 94, 0.3);' : 
                'background: rgba(239, 68, 68, 0.1); color: rgb(239, 68, 68); border: 1px solid rgba(239, 68, 68, 0.3);'
            }
        `;
        
        feedback.innerHTML = correct ? 
            `✅ Correct! "${answer}" is the right answer.` : 
            `❌ Not quite right. Try again!`;

        container.appendChild(feedback);
    }

    setupThemeToggle() {
        // Enhance the existing Furo theme toggle
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                // Save theme preference
                const theme = document.documentElement.dataset.theme;
                localStorage.setItem('sensor_fusion_theme', theme);
            });
        }

        // Restore theme preference
        const savedTheme = localStorage.getItem('sensor_fusion_theme');
        if (savedTheme) {
            document.documentElement.dataset.theme = savedTheme;
        }
    }

    setupMobileNavigation() {
        // Enhance mobile navigation experience
        let lastScrollTop = 0;
        const header = document.querySelector('.bd-header');
        
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            if (scrollTop > lastScrollTop && scrollTop > 100) {
                // Scrolling down
                if (header) header.style.transform = 'translateY(-100%)';
                const tracker = document.getElementById('progress-tracker');
                if (tracker && !tracker.classList.contains('hidden')) {
                    tracker.style.transform = 'translateX(270px)';
                }
            } else {
                // Scrolling up
                if (header) header.style.transform = 'translateY(0)';
                const tracker = document.getElementById('progress-tracker');
                if (tracker && !tracker.classList.contains('hidden')) {
                    tracker.style.transform = 'translateX(0)';
                }
            }
            
            lastScrollTop = scrollTop;
        });
    }

    // Method to manually mark exercises as complete
    markExerciseComplete(exerciseId) {
        const exerciseKey = `exercise_${exerciseId}`;
        if (!this.progress.completedQuizzes.includes(exerciseKey)) {
            this.progress.completedQuizzes.push(exerciseKey);
            this.saveProgress();
            this.updateProgressDisplay();
        }
    }

    // Export progress for debugging or analytics
    exportProgress() {
        return {
            ...this.progress,
            totalVisited: this.progress.visitedPages.length,
            totalQuizzes: this.progress.completedQuizzes.length,
            export_date: new Date().toISOString()
        };
    }

    // Reset progress (for testing or fresh start)
    resetProgress() {
        if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
            localStorage.removeItem(this.storageKey);
            localStorage.removeItem('sensor_fusion_theme');
            this.progress = {
                visitedPages: [],
                completedQuizzes: [],
                currentCourse: null,
                lastVisited: null
            };
            this.updateProgressDisplay();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.courseProgress = new CourseProgress();
    
    // Add global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + / to toggle progress tracker
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            const tracker = document.getElementById('progress-tracker');
            if (tracker) {
                document.getElementById('toggle-tracker').click();
            }
        }
        
        // Ctrl/Cmd + R to show progress reset option (for development)
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
            e.preventDefault();
            console.log('Progress:', window.courseProgress.exportProgress());
            if (confirm('Show reset option?')) {
                window.courseProgress.resetProgress();
            }
        }
    });
});

// Utility functions for content creators
window.sensorFusionUtils = {
    markExerciseComplete: (id) => window.courseProgress?.markExerciseComplete(id),
    getProgress: () => window.courseProgress?.exportProgress(),
    resetProgress: () => window.courseProgress?.resetProgress()
};