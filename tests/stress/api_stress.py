from locust import HttpUser, task


class StressUser(HttpUser):

    @task
    def predict_results_default(self):
        self.client.get(
            "/pipeline/predict/results",
            params={"page": 1, "page_size": 10},
        )

    @task
    def predict_results_paged(self):
        self.client.get(
            "/pipeline/predict/results",
            params={"page": 2, "page_size": 5},
        )
