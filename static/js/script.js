document.addEventListener('DOMContentLoaded', function() {
    // Animation du fond
    const background = document.querySelector('.background-animation');
    let mouseX = 0;
    let mouseY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX / window.innerWidth;
        mouseY = e.clientY / window.innerHeight;
        
        background.style.transform = `translate(${mouseX * 20}px, ${mouseY * 20}px)`;
    });

    // Animation des cartes de descripteurs
    const descriptorCards = document.querySelectorAll('.descriptor-card');
    descriptorCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'scale(1.05) translateY(-5px)';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'scale(1) translateY(0)';
        });
    });

    // Animation du bouton de soumission
    const submitBtn = document.querySelector('.submit-btn');
    if (submitBtn) {
        submitBtn.addEventListener('mouseenter', () => {
            submitBtn.style.transform = 'translateY(-2px)';
            submitBtn.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        });

        submitBtn.addEventListener('mouseleave', () => {
            submitBtn.style.transform = 'translateY(0)';
            submitBtn.style.boxShadow = 'none';
        });
    }

    // Animation des inputs
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.style.transform = 'scale(1.02)';
        });

        input.addEventListener('blur', () => {
            input.parentElement.style.transform = 'scale(1)';
        });
    });

    // Gestion des messages flash
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            message.style.transform = 'translateX(100%)';
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });

    // Validation des poids des descripteurs
    const weightInputs = document.querySelectorAll('.weight-input');
    weightInputs.forEach(input => {
        input.addEventListener('input', () => {
            const value = parseFloat(input.value);
            if (value < 0) {
                input.value = 0;
            }
        });
    });

    // Animation des sections du formulaire
    const formSections = document.querySelectorAll('.form-section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });

    formSections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'all 0.5s ease-out';
        observer.observe(section);
    });

    // Gestion du drag & drop pour l'image requête
    const queryImageInput = document.querySelector('#query_image');
    const inputGroup = queryImageInput.parentElement;

    inputGroup.addEventListener('dragover', (e) => {
        e.preventDefault();
        inputGroup.style.borderColor = 'var(--primary-color)';
        inputGroup.style.backgroundColor = 'rgba(74, 144, 226, 0.1)';
    });

    inputGroup.addEventListener('dragleave', () => {
        inputGroup.style.borderColor = '';
        inputGroup.style.backgroundColor = '';
    });

    inputGroup.addEventListener('drop', (e) => {
        e.preventDefault();
        inputGroup.style.borderColor = '';
        inputGroup.style.backgroundColor = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            queryImageInput.files = files;
        }
    });
}); 