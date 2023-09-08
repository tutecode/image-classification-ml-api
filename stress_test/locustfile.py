from locust import HttpUser, task, between


class APIUser(HttpUser):
    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    wait_time = between(1, 5)

    @task(1)
    def index(self):
        # self.client.get("/")
        self.client.get("http://127.0.0.1/")  # localhost

    @task(5)
    def predict(self):
        files = [("file", ("dog.jpeg", open("dog.jpeg", "rb"), "image/jpeg"))]
        headers = {}
        payload = {}
        # self.client.post("/predict",files={'file': open("../tests/dog.jpeg", "rb")})
        # self.client.post("/predict",
        self.client.post(
            "http://127.0.0.1/predict", headers=headers, data=payload, files=files
        )

    # raise NotImplementedError
