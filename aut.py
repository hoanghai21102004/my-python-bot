import requests
import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext


def fetch_historical_results(limit=100):
    url = "https://m.coinvid.com/api/rocket-api/game/issue-result/page"
    params = {"subServiceCode": "RG1M", "size": limit}

    headers = {
        "Host": "m.coinvid.com",
        "Connection": "keep-alive",
        "sec-ch-ua": '"Chromium";v="124", "Android WebView";v="124", "Not-A.Brand";v="99"',
        "user_type": "rocket",
        "Blade-Auth": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJpc3N1c2VyIiwiYXVkIjoiYXVkaWVuY2UiLCJ0ZW5hbnRfaWQiOiI2NzczNDMiLCJ1c2VyX25hbWUiOiJoYWlhbmhuZTI2IiwidG9rZW5fdHlwZSI6ImFjY2Vzc190b2tlbiIsInJvbGVfbmFtZSI6IiIsInVzZXJfdHlwZSI6InJvY2tldCIsInVzZXJfaWQiOiIxNjI0NzQ4OTg5OTY5NDA4MDAxIiwiZGV0YWlsIjp7ImF2YXRhciI6IjIwIiwidmlwTGV2ZWwiOjJ9LCJhY2NvdW50IjoiaGFpYW5obmUyNiIsImNsaWVudF9pZCI6InJvY2tldF93ZWIiLCJleHAiOjE3MTgzNzc4MzcsIm5iZiI6MTcxNzc3MzAzN30.Vr6idkzEk3gGaAwmvhCBs4bHudP2IAgaGkasthFVH-tG3GjPKwcvW0Yr3hupTQocu1i49zv7Beq7fqQepHcAOg",
        "Accept-Language": "en-US",
        "sec-ch-ua-mobile": "?1",
        "Authorization": "Basic cm9ja2V0X3dlYjpyb2NrZXRfd2Vi",
        "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "sec-ch-ua-platform": "Android",
        "X-Requested-With": "mark.via.gp",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://m.coinvid.com/game/guessMain?gameName=RG1M&returnUrl=%2FgameList",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Cookie": "_fbp=fb.1.1717773039503.427590769630604432; JSESSIONID=J_gUYvXcOABIiKF8uSbuy1Yh06ZK5PqbklIwzXbD"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'records' in data['data']:
            return data['data']['records']
    return []


def preprocess_data(records):
    df = pd.DataFrame(records)
    df['color'] = df['simpleResultFormatList'].apply(lambda x: x[0].get('color') if x else None)
    df['color'] = df['color'].map({'green': 0, 'red': 1})
    df['issue'] = pd.to_numeric(df['issue'], errors='coerce')
    df = df.dropna(subset=['issue', 'color'])

    # Thêm các tính năng mới
    df['previous_color'] = df['color'].shift(-1)
    df['color_change'] = (df['color'] != df['previous_color']).astype(int)
    df['issue_diff'] = df['issue'].diff().fillna(0)

    df = df.dropna(subset=['previous_color'])

    return df[['issue', 'color', 'previous_color', 'color_change', 'issue_diff']]


def train_model(df):
    X = df[['issue', 'previous_color', 'color_change', 'issue_diff']]
    y = df['color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return model


def predict_next_result(model, next_issue, last_color, last_color_change, last_issue_diff):
    prediction = model.predict([[next_issue, last_color, last_color_change, last_issue_diff]])
    return 'XANH' if prediction[0] == 0 else 'ĐỎ'


async def start(update: Update, context: CallbackContext) -> None:
    context.user_data['last_issue'] = None
    context.user_data['last_color'] = None
    context.user_data['last_color_change'] = None
    context.user_data['last_issue_diff'] = None
    context.user_data['model'] = None
    context.user_data['predictions_count'] = 0
    context.user_data['correct_predictions'] = 0
    context.user_data['running'] = True
    context.user_data['result_printed'] = False  # Flag to track if the result is printed

    chaohoi = """────────────────────────────────────────
=>TOOL DỰ ĐOÁN PHIÊN XANH ĐỎ COINVID VER 1.0
=>ADMIN: HOANG HAI X HAI ANH
────────────────────────────────────────"""

    await update.message.reply_text(chaohoi)
    time.sleep(2)
    await update.message.reply_text("『Bot đã hoạt động, bắt đầu dự đoán phiên...』")


    while context.user_data['running']:
        try:
            historical_results = fetch_historical_results(limit=100)
            if historical_results:
                df = preprocess_data(historical_results)
                last_result = df.iloc[0]
                issue = last_result['issue']
                color = 'XANH' if last_result['color'] == 0 else 'ĐỎ'
                color_change = last_result['color_change']
                issue_diff = last_result['issue_diff']

                if context.user_data['last_issue'] is None or issue != context.user_data['last_issue']:
                    if not context.user_data['result_printed']:
                        await update.message.reply_text(f"🙀 Phiên trước: Phiên : {issue} | Kết quả : {color}")
                        context.user_data['result_printed'] = True

                    if context.user_data['model'] is None:
                        context.user_data['model'] = train_model(df)

                    if context.user_data['last_issue'] is not None and issue != context.user_data['last_issue']:
                        next_issue = context.user_data['last_issue'] + 1
                        next_prediction = predict_next_result(context.user_data['model'], next_issue,
                                                              context.user_data['last_color'],
                                                              context.user_data['last_color_change'],
                                                              context.user_data['last_issue_diff'])
                        await update.message.reply_text("🥷 Dự đoán phiên tiếp theo: " + next_prediction)

                        if next_prediction == color:
                            context.user_data['correct_predictions'] += 1
                        context.user_data['predictions_count'] += 1

                        if context.user_data['predictions_count'] == 10:
                            accuracy = context.user_data['correct_predictions'] / context.user_data['predictions_count']
                            await update.message.reply_text(f"Tổng số lệnh thắng và thua sau 10 phiên là: {accuracy:.2f}")
                            context.user_data['predictions_count'] = 0
                            context.user_data['correct_predictions'] = 0

                    context.user_data['last_issue'] = issue
                    context.user_data['last_color'] = last_result['color']
                    context.user_data['last_color_change'] = color_change
                    context.user_data['last_issue_diff'] = issue_diff

                else:
                    context.user_data['result_printed'] = False

            else:
                await update.message.reply_text("Không có kết quả lịch sử để phân tích.")

        except Exception as e:
            await update.message.reply_text(f"Lỗi: {e}")

        time.sleep(5)



async def stop(update: Update, context: CallbackContext) -> None:
    context.user_data['running'] = False
    await update.message.reply_text("Bot stopped!")


def main():
    application = Application.builder().token("7100769571:AAGe3JMkpL62-2ffKtM7dLenDnx6DXDNZBk").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    application.run_polling()


if __name__ == "__main__":
    main()
