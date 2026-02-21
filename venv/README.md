\# \*\*ezvision – Drone Analytics Solution Accelerator\*\*



`ezvision` is a modular, end‑to‑end framework for building \*\*AI‑powered aerial vision analytics\*\*, integrating vector search, grounding, ETL pipelines, UI components and solution accelerators.  

This repository contains the \*\*reference implementation\*\* of the ezvision stack, including API services, ETL workflows, UI modules, and reusable solution accelerators.



The `venv/` directory (visible in this repo) organizes the system into four major components:



```

venv/

├── my\_droneworld\_api

├── my\_droneworld\_etl

├── my\_droneworld\_ui

└── solution\_accelerator

```



Each module is self‑contained and designed for reproducible benchmarking, rapid prototyping, and integration into production‑grade drone analytics pipelines.



---



\## 🚀 \*\*Key Features\*\*



\### \*\*1. Modular Architecture\*\*

\- Clean separation of API, ETL, UI, and accelerator logic  

\- Enables independent development, testing, and deployment  

\- Supports cloud‑native and edge‑augmented workflows



\### \*\*2. Vector Search–Driven Analytics\*\*

\- Built‑in support for vector store ingestion, export, and import  

\- Optimized for drone imagery, geospatial embeddings, and multimodal retrieval



\### \*\*3. End‑to‑End Droneworld Workflow\*\*

\- API for ingesting drone imagery and metadata  

\- ETL pipelines for preprocessing, tiling, and embedding  

\- UI for operator‑assistive visualization  

\- Solution accelerator for rapid deployment and benchmarking



\### \*\*4. Reproducible Benchmarking\*\*

\- Designed to integrate with our ezbenchmark framework  

\- Supports evaluation of vision‑LLMs, object detection models, and retrieval pipelines  

\- Ideal for research, publication, and enterprise validation



---



\## 📦 \*\*Repository Structure\*\*



| Directory | Description |

|----------|-------------|

| \*\*`my\_droneworld\_api/`\*\* | API layer for ingesting drone data, serving embeddings, and orchestrating analytics workflows. |

| \*\*`my\_droneworld\_etl/`\*\* | ETL pipelines for preprocessing, vectorization, and dataset management. Includes export/import of vector store data. |

| \*\*`my\_droneworld\_ui/`\*\* | Lightweight UI for visualizing droneworld analytics, operator workflows, and retrieval results. |

| \*\*`solution\_accelerator/`\*\* | Reusable templates, scripts, and workflows for deploying drone analytics solutions quickly. |



(Directory names and commit messages sourced from the GitHub tree view   \[github.com](https://github.com/ravibeta/ezvision/tree/main/venv).)



---



\## 🛠️ \*\*Installation\*\*



Clone the repository:



```bash

git clone https://github.com/ravibeta/ezvision.git

cd ezvision

```



Install dependencies (recommended: Python 3.10+):



```bash

pip install -r requirements.txt

```



If modules require local virtual environments, activate them per component:



```bash

cd venv/my\_droneworld\_api

source venv/bin/activate

```



---



\## ▶️ \*\*Usage\*\*



\### \*\*Run the API\*\*

```bash

cd venv/my\_droneworld\_api

python manage.py runserver

```



\### \*\*Run ETL Pipelines\*\*

```bash

cd venv/my\_droneworld\_etl

run shell scripts or python

```



\### \*\*Launch the UI\*\*

```bash

cd venv/my\_droneworld\_ui

npm install

npm start

```



\### \*\*Use the Solution Accelerator\*\*

```bash

cd venv/solution\_accelerator

tf plan

tf apply

```



---



\## 📊 \*\*Benchmarking \& Research\*\*



`ezvision` is designed to pair with our \*\*ezbenchmark\*\* framework for:



\- evaluating drone vision models  

\- comparing retrieval pipelines  

\- integrating vision‑LLMs  

\- generating publication‑grade metrics and visualizations  



This repository provides the operational backbone for those experiments.

ezbenchmark is available at https://github.com/ravibeta/ezbenchmark



---



\## 🤝 \*\*Contributing\*\*



Contributions are welcome!  

Please open an issue or submit a pull request with clear descriptions and reproducible examples.



---



\## 📄 \*\*License\*\*



This project is licensed under the terms of the repository’s `LICENSE` file.





