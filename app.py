from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:
        
        player1_name = request.form.get("player1_name")
        player2_name = request.form.get("player2_name")

        data = CustomData(

            court=request.form.get("court"),
            surface=request.form.get("surface"),
            round=request.form.get("round"),
            best_of=int(request.form.get("best_of")),

            h2h_winpct=float(request.form.get("h2h_winpct")),

            recentwinpct_1=float(request.form.get("recentwinpct_1")),
            recentwinpct_2=float(request.form.get("recentwinpct_2")),

            season=request.form.get("season"),

            surface_winpct_1=float(request.form.get("surface_winpct_1")),
            surface_winpct_2=float(request.form.get("surface_winpct_2")),

            higher_rank_player=int(request.form.get("higher_rank_player")),

            abs_rank_diff=int(request.form.get("abs_rank_diff"))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        winner = player1_name if result[0] == 1 else player2_name

        return render_template(
            "home.html",
            results=winner,
            player1_name=player1_name,
            player2_name=player2_name
        )


if __name__ == "__main__":\
    app.run(host="0.0.0.0", debug=True)