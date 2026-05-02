CREATE TABLE medquad (
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer TEXT,
    embedding BLOB -- This is where the vector lives
);