# 📝 AI Agent Blog Writing Platform

A production-ready, containerized multi-agent blog generation platform. Built with **Streamlit** for the frontend, **LangGraph** for orchestrating autonomous research and writing agents, and **Docker** for seamless local and cloud deployments.

This project features a secure OTP-based user registration and admin approval workflow, a persistent user workspace, automated image generation, and a human-in-the-loop refinement interface.

---

## 🚀 Key Features

*   **Multi-Agent LangGraph Pipeline**: Orchestrates specialised research, orchestration, content generation, and image placement agents.
*   **OTP Passwordless Authentication**: Secure login using one-time passcodes sent via email, featuring:
    *   Instant admin bypass configuration.
    *   **Admin Dashboard** for approving/rejecting new access requests.
    *   **Live status polling** ("Check Approval Status" button) for pending users.
*   **Aesthetic User Workspace**: Sleek responsive design utilizing custom CSS styling, welcoming users by name, and displaying a history of their past generated blogs.
*   **Refinement Feedback Loop**: Allows users to rate generated blogs, write comments, and request real-time modifications before saving.
*   **Dockerized Stack**: Simple cross-platform setup using Docker Compose with persistent data volumes for user sessions and assets.

---

## 🛠️ Tech Stack

*   **Frontend**: Streamlit, Custom CSS
*   **Orchestration & Agents**: LangGraph, LangChain
*   **AI Integration**: OpenAI (GPT models), Tavily Search (Research tool)
*   **Observability & Tracing**: LangSmith
*   **Email Service**: SMTP
*   **Containerization**: Docker, Docker Compose
*   **Cloud Infrastructure**: AWS EC2, Elastic IP

---

## ⚙️ Project Structure

```
├── frontend.py           # Streamlit UI & authentication logic
├── backend.py            # LangGraph agent orchestration & LLM pipeline
├── backend2.py           # Alternate/backup backend workflow
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build specification
├── docker-compose.yml    # Multi-container service configuration
└── users/                # Persistent storage for user databases (JSON)
```

---

## 🌐 Live Application Access

This application is deployed and hosted live on AWS. To request the deployed URL and credentials for private testing/viewing, please contact:

✉️ **[khan.15@alumni.iitj.ac.in](mailto:khan.15@alumni.iitj.ac.in)**
