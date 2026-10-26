/**
 * PCOS AI — Interactive Frontend JavaScript
 * Handles drag-and-drop uploads, sample selection, live What-If simulation,
 * and Grad-CAM tab controls.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Drag & Drop Upload Handling
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("ultrasound_image");
    const previewContainer = document.getElementById("upload-preview");
    const previewImg = document.getElementById("preview-img");
    const uploadForm = document.getElementById("upload-form");
    const submitBtn = document.getElementById("submit-btn");

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        ["dragenter", "dragover"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelection(fileInput.files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files[0]) {
                handleFileSelection(fileInput.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        if (!file.type.match("image.*")) {
            alert("Please upload a valid image file (JPG, JPEG, PNG).");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (previewContainer) previewContainer.style.display = "block";
            const sampleInput = document.getElementById("sample_path_input");
            if (sampleInput) sampleInput.value = "";
            if (submitBtn) submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // 2. Sample Image Click Selection
    const sampleCards = document.querySelectorAll(".sample-card-btn");
    sampleCards.forEach(card => {
        card.addEventListener("click", () => {
            const samplePath = card.getAttribute("data-sample-path");
            const sampleInput = document.getElementById("sample_path_input");
            if (sampleInput) {
                sampleInput.value = samplePath;
                if (previewImg) previewImg.src = samplePath;
                if (previewContainer) previewContainer.style.display = "block";
                if (submitBtn) submitBtn.disabled = false;
                // Auto submit or focus
                uploadForm.submit();
            }
        });
    });

    // 3. What-If Lab Live Simulation
    const whatifForm = document.getElementById("whatif-form");
    if (whatifForm) {
        const inputs = whatifForm.querySelectorAll("input, select");
        inputs.forEach(input => {
            input.addEventListener("input", debounce(updateWhatIfSimulation, 300));
        });
    }

    function updateWhatIfSimulation() {
        if (!whatifForm) return;
        const formData = new FormData(whatifForm);
        const jsonBody = {};
        formData.forEach((val, key) => {
            jsonBody[key] = val;
        });

        fetch("/api/whatif", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(jsonBody)
        })
        .then(res => res.json())
        .then(data => {
            if (data.available) {
                // Update score displays
                const scoreElem = document.getElementById("sim-score-pct");
                const classElem = document.getElementById("sim-class-name");
                const fillElem = document.getElementById("sim-progress-fill");
                const deltaElem = document.getElementById("sim-delta");
                const tierElem = document.getElementById("sim-tier-badge");

                if (scoreElem) scoreElem.innerText = `${data.score_pct}%`;
                if (classElem) classElem.innerText = data.predicted_class;
                if (fillElem) {
                    fillElem.style.width = `${Math.min(100, data.score_pct)}%`;
                    fillElem.style.backgroundColor = data.risk_tier.color;
                }
                if (deltaElem) {
                    const sign = data.delta_pct >= 0 ? "+" : "";
                    deltaElem.innerText = `Baseline Shift: ${sign}${data.delta_pct}%`;
                    deltaElem.style.color = data.delta_pct > 0 ? "#DC2626" : "#059669";
                }
                if (tierElem) {
                    tierElem.innerText = `Tier ${data.risk_tier.tier}: ${data.risk_tier.category}`;
                    tierElem.style.backgroundColor = data.risk_tier.color;
                }

                // Update calculated BMI
                const bmiDisplay = document.getElementById("calc-bmi-display");
                if (bmiDisplay && data.active_parameters.bmi) {
                    bmiDisplay.innerText = data.active_parameters.bmi;
                }
            }
        })
        .catch(err => console.error("What-If simulation error:", err));
    }

    // Helper: Debounce
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
});
