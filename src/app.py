import httpx

input_data = [5.1, 3.5, 1.4, 0.2]  # Replace with your input data

url = "http://localhost:8000/predict/"
response = httpx.post(url, json=input_data)

if response.status_code == 200:
    prediction = response.json()["prediction"]
    print("Predicted class label:", prediction)
else:
    print("Failed to get prediction")
