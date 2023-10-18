// script.js
const audioElement = document.getElementById('audio-element');
const playButton = document.getElementById('play-button');
const progress = document.getElementById('progress');
const currentTimeDisplay = document.getElementById('current-time');
const durationDisplay = document.getElementById('duration');

let isPlaying = false;

playButton.addEventListener('click', toggleAudio);

audioElement.addEventListener('timeupdate', updateProgressBar);
audioElement.addEventListener('loadedmetadata', updateDuration);

function toggleAudio() {
    if (isPlaying) {
        
        audioElement.pause();
        playButton.style.backgroundImage = 'static/play.jpg';
        
    } else {
        audioElement.play();
        playButton.style.backgroundImage = 'static/play.jpg';
    }
    isPlaying = !isPlaying;
}

function updateProgressBar() {
    const currentTime = audioElement.currentTime;
    const duration = audioElement.duration;
    const progressPercentage = (currentTime / duration) * 100;
    
    progress.style.width = progressPercentage + '%';

    currentTimeDisplay.textContent = formatTime(currentTime);
}

function updateDuration() {
    const duration = audioElement.duration;
    durationDisplay.textContent = formatTime(duration);
}

function formatTime(time) {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return minutes + ':' + (seconds < 10 ? '0' : '') + seconds;
}
