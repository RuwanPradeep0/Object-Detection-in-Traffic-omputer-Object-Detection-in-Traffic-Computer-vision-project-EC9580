document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.querySelector('input[type="file"]');
    const previewImage = document.createElement("img");
    const resultDiv = document.createElement("div");

    previewImage.style.maxWidth = "300px";
    previewImage.style.marginTop = "20px";
    resultDiv.style.marginTop = "20px";
    resultDiv.style.fontSize = "20px";

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                if (!document.body.contains(previewImage)) {
                    document.body.appendChild(previewImage);
                }
            };
            reader.readAsDataURL(file);
        }
    });

    const form = document.querySelector("form");
    form.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData(form);
        resultDiv.textContent = "Processing...";
        document.body.appendChild(resultDiv);

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.text())
            .then((result) => {
                resultDiv.textContent = `Prediction: ${result}`;
            })
            .catch((error) => {
                resultDiv.textContent = "Error in prediction. Please try again.";
                console.error("Error:", error);
            });
    });
});
