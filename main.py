import pyodbc
import pandas as pd
from sqlalchemy import create_engine
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def fetch_emails(username, password, imap_server, limit=10):
    with imaplib.IMAP4_SSL(imap_server) as mail:
        mail.login(username, password)
        mail.select('inbox')
        result, data = mail.search(None, 'NEW')
        email_ids = data[0].split()[-limit:]
        latest_emails = [email.message_from_bytes(mail.fetch(num, '(RFC822)')[1][0][1]) for num in email_ids]
        mail.logout()
    return latest_emails


def extract_body(email_msg):
    body = ""
    if email_msg.is_multipart():
        for part in email_msg.walk():
            if part.get("Content-Disposition") is None or "attachment" not in part.get("Content-Disposition"):
                payload = part.get_payload(decode=True)
                if payload:
                    part_body = decode_payload(part, payload)
                    if part.get_content_type() == "text/plain":
                        body += part_body
                    elif part.get_content_type() == "text/html" and not body:
                        soup = BeautifulSoup(part_body, "html.parser")
                        body += soup.get_text()
    else:
        payload = email_msg.get_payload(decode=True)
        if payload:
            body = decode_payload(email_msg, payload)
    return body.strip()


def decode_payload(part, payload):
    charset = part.get_content_charset() or 'utf-8'
    if isinstance(payload, bytes):
        return payload.decode(charset, errors="ignore")
    return payload


def decode_header_value(header_value):
    return ' '.join(
        part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part
        for part, encoding in decode_header(header_value)
    )


def remove_empty_lines(text):
    return '\n'.join(line for line in text.split('\n') if line.strip())


def display_email(email_msg):
    subject = decode_header_value(email_msg['Subject'])
    from_decoded = decode_header_value(email_msg['From'])
    to_decoded = decode_header_value(email_msg['To'])
    spam_status = email_msg.get('Spam', 'Nieznany')

    print("\n------------------------------------------")
    print(f"Tytuł: {subject} - Spam: {spam_status}")
    print(f"Od: {from_decoded}")
    print(f"Do: {to_decoded}")
    print("\nTreść wiadomości:")
    body = extract_body(email_msg)
    body_cleaned = remove_empty_lines(body)
    print(body_cleaned)
    print("------------------------------------------\n")


def clean_words(words):
    unwanted_patterns = [
        r"[.,!?;:\"()<>]",  # Punctuation
        r"\b(?:0px|table|none|nbsp|div|span)\b",  # Common HTML/CSS terms
        r"\b\w{1,2}\b"  # Single and double-letter words (optional)
    ]
    combined_pattern = '|'.join(unwanted_patterns)
    filtered_words = [word for word in words if not re.search(combined_pattern, word)]
    return filtered_words


def word_diversity(words):
    unique_words = set(words)
    diversity_ratio = len(unique_words) / len(words) if words else 0

    if diversity_ratio < 0.2:
        return "Bardzo mała"
    elif 0.2 <= diversity_ratio < 0.4:
        return "Mała"
    elif 0.4 <= diversity_ratio < 0.6:
        return "Średnia"
    elif 0.6 <= diversity_ratio < 0.8:
        return "Duża"
    else:
        return "Bardzo duża"


def email_statistics(emails):
    spam_bodies = []
    not_spam_bodies = []
    for email_msg in emails:
        body = extract_body(email_msg)
        if email_msg['Spam'] == 'Tak':
            spam_bodies.append(body)
        else:
            not_spam_bodies.append(body)

    def calculate_statistics(bodies):
        lengths = [len(body) for body in bodies]
        average_length = sum(lengths) / len(lengths) if lengths else 0
        all_words = ' '.join(bodies).split()
        clean_word_list = clean_words(all_words)
        word_counts = Counter(clean_word_list)
        most_common_words = word_counts.most_common(5)
        diversity = word_diversity(clean_word_list)
        return average_length, most_common_words, diversity

    spam_avg_length, spam_common_words, spam_diversity = calculate_statistics(spam_bodies)
    not_spam_avg_length, not_spam_common_words, not_spam_diversity = calculate_statistics(not_spam_bodies)

    print("\nStatystyki dla e-maili oznaczonych jako spam:")
    print(f"Średnia długość tekstu: {spam_avg_length:.2f}")
    print("5 najczęściej powtarzające się słowa:", ', '.join(f"{word} ({count})" for word, count in spam_common_words))
    print(f"Różnorodność słów: {spam_diversity}")

    print("\nStatystyki dla e-maili oznaczonych jako nie spam:")
    print(f"Średnia długość tekstu: {not_spam_avg_length:.2f}")
    print("5 najczęściej powtarzające się słowa:",
          ', '.join(f"{word} ({count})" for word, count in not_spam_common_words))
    print(f"Różnorodność słów: {not_spam_diversity}")

    plot_statistics(spam_avg_length, spam_common_words, spam_diversity, not_spam_avg_length, not_spam_common_words,
                    not_spam_diversity)


def plot_statistics(spam_avg_length, spam_common_words, spam_diversity, not_spam_avg_length, not_spam_common_words,
                    not_spam_diversity):
    labels = ['Spam', 'Nie spam']
    avg_lengths = [spam_avg_length, not_spam_avg_length]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, avg_lengths, color=['red', 'blue'])
    plt.title('Średnia długość tekstu w emailach')
    plt.ylabel('Długość tekstu')
    plt.show()

    spam_words, spam_counts = zip(*spam_common_words)
    not_spam_words, not_spam_counts = zip(*not_spam_common_words)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(spam_words, spam_counts, color='red')
    plt.title('Najczęściej powtarzające się słowa w spamie')
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    plt.bar(not_spam_words, not_spam_counts, color='blue')
    plt.title('Najczęściej powtarzające się słowa w nie spamie')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    diversity_labels = ['Bardzo mała', 'Mała', 'Średnia', 'Duża', 'Bardzo duża']
    spam_diversity_index = diversity_labels.index(spam_diversity)
    not_spam_diversity_index = diversity_labels.index(not_spam_diversity)

    plt.figure(figsize=(8, 5))
    plt.bar(['Spam', 'Nie spam'], [spam_diversity_index, not_spam_diversity_index], color=['red', 'blue'])
    plt.title('Różnorodność słów w emailach')
    plt.ylabel('Stopień różnorodności')
    plt.yticks(range(len(diversity_labels)), diversity_labels)
    plt.show()


def classify_emails(emails, vectorizer, clf):
    for email_msg in emails:
        body = extract_body(email_msg)
        if body.strip():
            X_new = vectorizer.transform([body])
            email_msg['Spam'] = "Tak" if clf.predict(X_new)[0] == 1 else "Nie"
    return emails


def main():
    username = 'machinelearning@o2.pl'
    password = 'M75$$uyt(SAeiz$'
    imap_server = 'poczta.o2.pl'

    emails = fetch_emails(username, password, imap_server, limit=5)

    server = 'your-server.database.windows.net'
    database = 'your-database'
    username = 'your-username'
    password = 'your-password'
    driver = '{ODBC Driver 17 for SQL Server}'

    connection_string = f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}?driver={driver}"
    engine = create_engine(connection_string)

    query = "SELECT Body, Label FROM SpamData"
    spam_data = pd.read_sql(query, engine)
    spam_data['Body'] = spam_data['Body'].fillna('')
    texts, labels = spam_data["Body"], spam_data["Label"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = MultinomialNB()
    clf.fit(X, labels)

    emails = classify_emails(emails, vectorizer, clf)

    for idx, email_msg in enumerate(emails):
        subject_decoded = decode_header_value(email_msg['Subject'])
        spam_status = email_msg.get('Spam', 'Nieznany')
        print(f"{idx + 1}. Tytuł: {subject_decoded} - Spam: {spam_status}")

    while True:
        try:
            choice = int(input("Wybierz numer wiadomości, którą chcesz zobaczyć (lub wpisz 0 aby zakończyć): "))
            if choice == 0:
                break
            elif 1 <= choice <= len(emails):
                display_email(emails[choice - 1])
            else:
                print("Nieprawidłowy numer wiadomości. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowe wejście. Podaj numer wiadomości jako liczbę całkowitą.")

    email_statistics(emails)


if __name__ == "__main__":
    main()
