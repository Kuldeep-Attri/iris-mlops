import httpx
import streamlit as st


def get_prediction(input_data):
    """
    Get prediction from the prediction server.

    Args:
        input_data (list): Input data for prediction.

    Returns:
        str: Predicted class label if successful, or an error message if failed.
    """
    url = "http://127.0.0.1:8000/predict/"

    try:
        # Send a POST request to the prediction server
        response = httpx.post(url, json=input_data)

        if response.status_code == 200:
            prediction = response.json().get("prediction")
            return f"{prediction}"
        else:
            return "Failed to get prediction"
    except httpx.RequestError as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    """
    Main function to create the Streamlit user interface.
    """

    st.title("Iris Flower Classification")

    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width = st.number_input("Sepal Width", value=3.5)
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width = st.number_input("Petal Width", value=0.2)

    input_data = [sepal_length, sepal_width, petal_length, petal_width]

    if st.button("Predict"):
        prediction_result = get_prediction(input_data)
        mapping_ = {
            "0": "Iris-setosa",
            "1": "Iris-versicolor",
            "2": "Iris-virginica",
        }
        st.write(f"Output is: {mapping_[prediction_result]}.")


if __name__ == "__main__":
    main()
