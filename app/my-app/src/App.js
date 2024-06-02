import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <div class="container">
        <div class="in">
          <form action="http://127.0.0.1:8000/" method="post">
            <input
              class="in_text"
              type="text"
              name="client_str"
              id="client_str"
              placeholder="긍정인지 부정인지 궁금하신가요?"
            />
            <input class="in_btn" type="submit" value="전송" />
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
