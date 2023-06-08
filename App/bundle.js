import React from 'react'
import ReactDOM from 'react-dom'

function App(){
    return (
        <div>
            <h1>This is the web App</h1>
            <p>This text was made from the react script</p>
        </div>
    );
}

ReactDOM.render(<App />,document.getElementById("root"));