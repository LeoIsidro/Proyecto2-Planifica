from flask import Flask, render_template
from flasgger import Swagger


def create_app():
    app = Flask(__name__)
    app.config["SWAGGER"] = {"title": "SYNTH_LAB API", "uiversion": 3}
    Swagger(app, template={"info": {"title": "SYNTH_LAB API", "version": "1.0"}})

    from .routes import bp
    app.register_blueprint(bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    return app
