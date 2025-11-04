// When button is clicked
document.querySelector(".demo-btn").addEventListener("click", function () {
  const inputValue = document.querySelector(".demo-i").value.trim();

  // Hide all state card groups first
  const allStates = [
    "kerala",
    "tamil",
    "karnataka",
    "andra",
    "jharkhand",
    "gujarat",
  ];
  allStates.forEach((state) => {
    document.querySelectorAll(`.${state}`).forEach((card) => {
      card.style.display = "none";
    });
  });

  // Decide which state's cards to show
  let stateClass = "";
  switch (inputValue) {
    case "1":
      stateClass = "kerala";
      break;
    case "2":
      stateClass = "tamil";
      break;
    case "3":
      stateClass = "karnataka";
      break;
    case "4":
      stateClass = "andra";
      break;
    case "5":
      stateClass = "jharkhand";
      break;
    case "6":
      stateClass = "gujarat";
      break;
    default:
      alert("Enter a valid number (1-6)");
      return;
  }

  // Show selected state's cards
  document.querySelectorAll(`.${stateClass}`).forEach((card) => {
    card.style.display = "block";
  });

  // Optionally update "Your state is:" text
  const stateNameMap = {
    kerala: "Kerala",
    tamil: "Tamil Nadu",
    karnataka: "Karnataka",
    andra: "Andhra Pradesh",
    jharkhand: "Jharkhand",
    gujarat: "Gujarat",
  };
  document.querySelector(".state").textContent = stateNameMap[stateClass];
  document.querySelector(".state").style.display = "inline";
});