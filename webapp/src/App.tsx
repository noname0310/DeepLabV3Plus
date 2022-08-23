import styled from "styled-components";

const OuterDiv = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    width: 100vw;
`;

function App(): JSX.Element {
    return (
        <OuterDiv>
            <h1>Hello World</h1>
        </OuterDiv>
    );
}

export default App;
