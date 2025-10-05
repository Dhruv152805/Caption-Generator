import requests

data = {
    "url": "https://www.youtube.com/watch?v=nvmOhpuIhfI",
    "desired_sentences": 6
}

response = requests.post("http://127.0.0.1:8000/summarize", json=data)
print(response.json())
