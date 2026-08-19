import { useState } from "react";
import "./App.css";

type Source = {
  title: string;
  source: string;
  url: string;
  score: number;
};

type ApiResponse = {
  question: string;
  answer: string;
  sources: Source[];
};

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
          errorData?.detail || "Something went wrong."
        );
      }

      const data: ApiResponse = await response.json();

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    askQuestion();
  };

  return (
    <main className="app">
      <section className="hero-section">
        <div className="badge">
          Evidence-Grounded Medical AI
        </div>

        <h1>
          Medical Report
          <span> Companion</span>
        </h1>

        <p className="subtitle">
          Ask questions about medical information and get
          evidence-grounded explanations from trusted sources.
        </p>

        <form className="question-form" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a medical question..."
            rows={4}
            disabled={loading}
          />

          <button
            type="submit"
            disabled={loading || !question.trim()}
          >
            {loading ? "Analyzing..." : "Ask Medical Companion"}
          </button>
        </form>

        <div className="examples">
          <span>Try:</span>

          <button
            type="button"
            onClick={() =>
              setQuestion("What does a high creatinine level mean?")
            }
          >
            High creatinine
          </button>

          <button
            type="button"
            onClick={() =>
              setQuestion("What is HbA1c used for?")
            }
          >
            HbA1c
          </button>

          <button
            type="button"
            onClick={() =>
              setQuestion("What does TSH measure?")
            }
          >
            TSH
          </button>
        </div>
      </section>

      {error && (
        <section className="error-card">
          <strong>Unable to answer</strong>
          <p>{error}</p>

          <small>
            Make sure the FastAPI backend is running on
            http://127.0.0.1:8000
          </small>
        </section>
      )}

      {loading && (
        <section className="loading-card">
          <div className="loader"></div>

          <div>
            <strong>Analyzing your question...</strong>
            <p>
              Searching the medical knowledge base and
              generating an evidence-grounded answer.
            </p>
          </div>
        </section>
      )}

      {result && !loading && (
        <section className="result-section">
          <div className="result-card">
            <div className="result-header">
              <div>
                <span className="result-label">
                  Your Question
                </span>

                <h2>{result.question}</h2>
              </div>

              <span className="verified-badge">
                Evidence Grounded
              </span>
            </div>

            <div className="answer">
              {result.answer}
            </div>
          </div>

          <div className="sources-section">
            <div className="section-heading">
              <div>
                <span className="result-label">
                  Retrieved Evidence
                </span>

                <h2>Sources</h2>
              </div>

              <span className="source-count">
                {result.sources.length} source
                {result.sources.length !== 1 ? "s" : ""}
              </span>
            </div>

            {result.sources.length === 0 ? (
              <div className="no-sources">
                No sources were returned.
              </div>
            ) : (
              <div className="sources-list">
                {result.sources.map((source, index) => (
                  <article
                    className="source-card"
                    key={`${source.url}-${index}`}
                  >
                    <div className="source-number">
                      {index + 1}
                    </div>

                    <div className="source-content">
                      <h3>{source.title}</h3>

                      <p className="source-name">
                        {source.source}
                      </p>

                      <div className="source-meta">
                        <span>
                          Retrieval score:{" "}
                          {source.score.toFixed(4)}
                        </span>

                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View source →
                        </a>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      <footer>
        <p>
          Medical Report Companion · Evidence-grounded
          medical information
        </p>

        <p>
          This tool provides information from the available
          sources and does not replace professional medical
          advice.
        </p>
      </footer>
    </main>
  );
}

export default App;