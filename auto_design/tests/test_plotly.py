from flask import Flask
from dash import Dash
import threading
import time
from wsgiref.simple_server import make_server

def stop_execution():
    global keepPlot
    #stream.stop_stream()
    keepPlot=False
    # stop the Flask server
    server.shutdown()
    server_thread.join()
    print("Dash app stopped gracefully.")
    
    
server = Flask(__name__)
app = Dash(__name__, server=server)


if __name__ == "__main__":
    # create a server instance
    server = make_server("localhost", 8050, server)
    # start the server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    # start the Dash app in a separate thread
    def start_dash_app():
        app.run_server(debug=True, use_reloader=False)

    dash_thread = threading.Thread(target=start_dash_app)
    dash_thread.start()

    for i in range(5):
        time.sleep(1)  # keep the main thread alive while the other threads are running
    
    stop_execution()