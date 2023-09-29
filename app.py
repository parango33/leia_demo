from flask import Flask, render_template, request, jsonify
from leia_engine import parse_document

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    cliente = request.form["cliente"]
    abogado = request.form["abogado"]
    tipo_documento = request.form["tipo_doc"]
    objetivo_documento = request.form["objetivo_doc"]
    doc_info= parse_document(doc_indicator=tipo_documento, n1=cliente, n2=abogado, objective=objetivo_documento)

    return jsonify(
        {
            "summary": doc_info.summary,
            "generated_document": doc_info.generated_document
        }
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0')



