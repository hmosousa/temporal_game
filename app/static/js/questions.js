document.addEventListener('DOMContentLoaded', () => {
    const answerButtons = document.querySelectorAll('.relation-button');
    const feedbackDiv = document.getElementById('feedback');
    const feedbackMessage = document.getElementById('feedback-message');
    const correctAnswerSpan = document.getElementById('correct-answer');

    answerButtons.forEach(button => {
        button.addEventListener('click', () => {
            const answer = button.dataset.relation;

            // Disable all buttons after selection
            answerButtons.forEach(btn => btn.style.pointerEvents = 'none');

            // Send the answer to the server
            fetch('/api/submit_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    answer: answer, 
                    question_idx: currentIdx,
                    user_id: userId
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Show feedback
                const isCorrect = answer === data.correct_answer;
                feedbackMessage.className = isCorrect ? 'feedback-correct' : 'feedback-incorrect';
                feedbackMessage.textContent = isCorrect ? 
                    '✓ Correct!' : 
                    '✗ Incorrect';
                
                correctAnswerSpan.textContent = data.correct_answer;
                feedbackDiv.style.display = 'block';

                // Highlight the correct and incorrect buttons
                answerButtons.forEach(btn => {
                    if (btn.dataset.relation === data.correct_answer) {
                        btn.classList.add('correct-answer');
                    }
                    if (btn.dataset.relation === answer && !isCorrect) {
                        btn.classList.add('incorrect-answer');
                    }
                });
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    });
}); 