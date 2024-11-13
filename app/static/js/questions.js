document.addEventListener('DOMContentLoaded', () => {
    const answerButtons = document.querySelectorAll('.relation-button');

    answerButtons.forEach(button => {
        button.addEventListener('click', () => {
            const answer = button.dataset.answer;

            // Send the answer to the server
            fetch('/api/submit_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ answer: answer, question: questionData }),
            })
            .then(response => response.json())
            .then(data => {
                alert('Your answer has been submitted!');
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    });
}); 