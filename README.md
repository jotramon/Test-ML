# MLE Challenge — Flight Delay Prediction on GCP

## Context

A Data Scientist at LATAM explored flight delay prediction for Santiago (SCL) airport in `challenge/exploration.ipynb`. The notebook works well as an experiment, but it's not production-ready.

Your job is to productionize this work into a cloud-native ML system on GCP: from raw CSV data to automated training and batch inference pipelines, orchestrated by an API and deployed continuously via GitHub Actions.

---

## Repository setup

> **Important!**
> - [**Github free for personal account**](https://docs.github.com/en/get-started/learning-about-github/githubs-plans?utm_source=chatgpt.com)  is enough for the challenge completeness, no payment is required to use the github features that are required in this challenge.

1. Create a **private repository** in **github.com** 
2. [Invite the evaluators](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository?utm_source=chatgpt.com) to your private repository. Evaluators account can be found on [TBD](TBD)

3. Use the **main** branch for any official release that we should review. It is highly recommended to use [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) development practices. **NOTE: do not delete your development branches.**
   
4. Please, do not change the structure of the challenge (names of folders and files).
   
4. All the documentation and explanations that you have to give us must go in the `challenge.md` file inside `docs` folder.

5. To send your challenge, you must do a `POST` request to:
    `https://advana-challenge-check-api-cr-k4hdbggvoq-uc.a.run.app/software-engineer`
    This is an example of the `body` you must send:
    ```json
    {
      "name": "Juan Perez",
      "mail": "juan.perez@example.com",
      "github_url": "https://github.com/juanperez/latam-challenge.git",
      "api_url": "https://juan-perez.api"
    }
    ```

***NOTE: We recommend to send the challenge even if you didn't manage to finish all the parts.***


---

## Part I — Model Implementation

Transcribe the notebook into the `DelayModel` class in `challenge/model.py`. The class must implement all the interfaces already defined in `challenge/model.py`.

- If you find any bug, fix it.
- The DS proposed a few models in the end. Choose the best model at your discretion, argue why. **It is not necessary to make improvements to the model.**
- Apply all the good programming practices that you consider necessary in this item.

> **Note:**
> - **You cannot** remove or change the name or arguments of **provided** methods.
> - **You can** extend the implementation of the provided methods if needed.
> - **You can** create the extra classes and methods you deem necessary.

**Acceptance criteria:** `make model-test` must pass, **you cannot change the unit test implementation**. The test suite enforces minimum recall and F1 thresholds on the delayed class — model selection and class imbalance handling matter.

---

## Part II — Data Infrastructure

> **Note**
> - Ensure you are using a single service account for the challenge deployment.

Before any pipeline can run, the raw data needs to be accessible in GCP.

- Upload training `data/data.csv` to a Cloud Storage bucket
- Upload serving `data/serving_input.csv` to a Cloud Storage bucket
- Load it each one into a BigQuery table that your pipelines will read from it.

You decide the bucket naming, table schema, and how to orchestrate the load. All infrastructure must be created via `gcloud` CLI, or IaC (terraform recommended) **no GCP web console allowed.**

> All infraestructure deployment commands and/or terraform modules must be in the root folder called `infra`.
> - For `gcloud` commands use the file called `infra/bootstrap.sh`
> - For the terraform modules use the folder `infra/terraform/*`

---

## Part III — Training Pipeline

Implement a training pipeline using the provided classes and deploy it via a cloud run job. The pipeline must live in the folder `/challenge/training.py`, and use the respective methods from the model class `DelayModel`.

How you structure the pipeline components/steps, pass artifacts between steps, is up to you.

---

## Part IV — Serving Pipeline

Implement a batch serving pipeline using the provided classes and deploy it via a cloud run job. The pipeline must live in the folder `/challenge/serving.py`, and use the respective methods from the model class `DelayModel`.

How you structure the pipeline components/steps, pass artifacts between steps, is up to you.

---

## Part V — Pipeline Control API

Deploy a FastAPI application that acts as a control plane for the two pipelines. The API must expose these endpoints:

| Method | Path | Behavior |
|--------|------|----------|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `POST` | `/pipeline/train` | Triggers the training pipeline synchronously; returns model metadata (version, metrics) once complete |
| `POST` | `/pipeline/predict` | Triggers the serving pipeline synchronously; returns the first 10 predictions once complete |
| `GET` | `/pipeline/predict/results` | Paginated read of the predictions table (query params: `page`, `page_size`, `opera`, `tipovuelo`, `mes` — optional comma-separated lists to filter results) |

**Acceptance criteria:** `make api-test` and `make stress-test` must pass.


> **Important**
> - The API must be available for testing during the challenge evaluation period, evaluators will test the deployed api endpoints.

---

## Part VI — Infrastructure & CI/CD

### GCP Resources

Provide a `gcloud`-based script or an IaC integrated pipeline that provisions all required infrastructure from scratch in a fresh GCP project. At minimum:

Terraform is a **nice-to-have** — if you provide it, it should be equivalent to the shell script.

---
### **CI/CD**

We are looking for a proper `CI/CD` implementation for this development.

- Complete both `ci.yml` and `cd.yml`(consider what you did in the previous parts).

---

## Constraints

- **Target cloud**: GCP (the $300 free trial credit covers the full challenge)
- **Infrastructure provisioning**: `gcloud` CLI is required; Terraform is optional but valued
- **CI/CD**: GitHub Actions is required
- **No web console**: every resource must be created and deployed via code or CLI

---

