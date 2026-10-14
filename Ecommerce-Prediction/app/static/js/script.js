document.addEventListener("DOMContentLoaded", function () {

    console.log("E-Commerce AI initialized.");

    /*
     * Small interaction:
     * automatically focus the first field
     * on the prediction page.
     */

    const firstInput =
        document.querySelector(
            ".prediction-form input"
        );

    if (firstInput) {
        firstInput.focus();
    }


    /*
     * Add a small loading state
     * when the prediction form is submitted.
     */

    const predictionForm =
        document.querySelector(
            ".prediction-form"
        );

    const predictButton =
        document.querySelector(
            ".predict-button"
        );


    if (predictionForm && predictButton) {

        predictionForm.addEventListener(
            "submit",
            function () {

                predictButton.disabled = true;

                predictButton.innerHTML =
                    "<span>◌</span>" +
                    " Analyzing Customer..." +
                    "<span>...</span>";

            }
        );

    }

});