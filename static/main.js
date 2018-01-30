// hello world

function onClickSubmit(){
  console.log("onClickSubmit");
  // todo: disable the input area,
    //__TODO__
  // parse context and question
  var context = document.getElementById("context_input").value;
  var question = document.getElementById("question_input").value;
  predict(context, question);
}

$(document).ready(function() {
  console.log($.ajax);
});

function predict(context, question){
  // send request to the ML service
  $.ajax({
    type: "POST",
    url: "/ask",
    data: { context: context, question: question },
    success: function( msg ) {
      var obj = JSON.parse(msg);
      var start_idx = obj['start_idx'];
      var end_idx = obj['end_idx']
      // show the result when recieved the response
      showAnswer(context, question, start_idx, end_idx)
    }
  });
}

function showAnswer(context, question, start_idx, end_idx){
  console.log("context", context);
  console.log("answer", context.substring(start_idx, end_idx));
  var answer_panel_placeholder = document.getElementById("answer_panel_placeholder");
  var answer_panel = document.getElementById("answer_panel");
  answer_panel_placeholder.addClass("d-none");
  answer_panel.removeClass("d-none");


  var answer_text = context.substring(start_idx, end_idx);
  var answer_text_after = context.substring(end_idx, context.length);
  var answer_text_before = context.substring(0, start_idx);

  document.getElementById("answer_question").innerHTML = question;

  var answer_summary = document.getElementById("answer_summary");
  answer_summary.innerHTML = answer_text;

  var e_answer_text  = document.getElementById("answer_text");
  var e_answer_text_after = document.getElementById("answer_text_after");
  var e_answer_text_before = document.getElementById("answer_text_before");

  e_answer_text.innerHTML = answer_text;
  e_answer_text_after.innerHTML = answer_text_after;
  e_answer_text_before .innerHTML = answer_text_before;

}

HTMLElement.prototype.addClass = function(add){
  console.log("add class function extended");
  var curClassName = this.className;
  if (curClassName.includes(add) == false){
    this.className = curClassName + " " + add;
  }
}

// a way to bake this functionality right into all DOM elements:
// https://stackoverflow.com/questions/2155737/remove-css-class-from-element-with-javascript-no-jquery
HTMLElement.prototype.removeClass = function(remove) {
  console.log("remove class function extended");
  var newClassName = "";
  var i;
  var classes = this.className.split(" ");
  for(i = 0; i < classes.length; i++) {
      if(classes[i] !== remove) {
          newClassName += classes[i] + " ";
      }
  }
  this.className = newClassName;
}
