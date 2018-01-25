-- create_db.sql
CREATE TABLE query_histories (
    id INTEGER PRIMARY KEY ASC AUTOINCREMENT,
    context TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_start INTEGER,
    answer_end INTEGER,
    record_time DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE valid (
    qa_id INTEGER,
    FOREIGN KEY (qa_id) REFERENCES query_histories(id)
);
