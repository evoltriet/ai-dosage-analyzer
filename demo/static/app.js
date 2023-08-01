angular.module('app', [])
.controller('notesController', function($scope, $http, $sce) {
  $scope.predictions = $sce.trustAsHtml('No Predictions Yet...')

  $scope.num_notes_to_process = 0;
  $scope.enter_note_active = false;
  $scope.entry = {};

  $scope.show_arch_diagram = false;

  $scope.show_arch = function(){
    $scope.show_arch_diagram = !$scope.show_arch_diagram;
    console.log($scope.show_arch_diagram);
  }

  $scope.notes_to_process = [
    ]
  $scope.notes_not_to_process = [
      {'note':'1 A DAY FOR BP'},
      {'note':'1 BD FOR ANAEMIA'},
      {'note':'1 CAPSULE PER DAY'},
      {'note':'1 DAILY TO TREAT UNDERACTIVE THYROID'},
      {'note':'USE ONE SPRAY THREE TIMES A DAY'},
      
      {'note':'1 EVERY MORNING'},
      {'note':'INHALE 2 DOSES FOUR TIMES DAILY'},
      {'note':'1 ONCE DAILY; FOR HIGH BLOOD PRESSURE'},
      {'note':'1 PUFF Q.D.S.'},
      {'note':'FIVE TO BE TAKEN DAILY'},
      
      {'note':'1 TAB EACH DAY'},
      {'note':'1 TABLET EACH DAY'},
      {'note':'1 TABLET TWICE DAILY'},
      {'note':'1 TABLET(S) TWICE A DAY.'},
      {'note':'1 THREE TIMES A DAY WITH MEALS OR AFTER FOOD'},
      
      {'note':'1 TWO TIMES PER DAY'},
      {'note':'INHALE 4 DOSES TWICE DAILY'},
      {'note':'2 DROPS 5 TIMES PER DAY'},
      {'note':'2 SACHETS DAILY'},
      {'note':'2 X5ML SPOON TWICE DAILY'},
      
      {'note':'4 DROPS QDS'},
      {'note':'6 TABLET ADAY'},
      {'note':'6 OD FOR 5 DAYS'},
      {'note':'3 THREE TIMES PER DAY'},
      {'note':'DISSOLVE CONTENTS OF ONE SACHET IN HALF A GLASS (125ML) OF WATER AND TAKE TWICE A DAY'},
      
      {'note':'1 FOUR TIMES A DAY WITH FOOD'},
      {'note':'8 EACH MORNING'},
      {'note':'1 EACH NIGHT'},
      {'note':'1 SPRAY INTO EACH NOSTRIL BD'},
      {'note':'1 EVERY DAY FOR BLOOD PRESSURE AND THE HEART, ALSO PROTECTS THE KIDNEYS IN DIABETES'}
       ]

  $scope.select_to_remove = function(index){
    var note = $scope.notes_to_process.splice(index, 1)[0];
    $scope.notes_not_to_process.push(note);

    $scope.num_notes_to_process = $scope.notes_to_process.length
  }

  $scope.select_to_add = function(index){
    var note = $scope.notes_not_to_process.splice(index, 1)[0];
    $scope.notes_to_process.push(note);

    $scope.num_notes_to_process = $scope.notes_to_process.length
  }

  $scope.clear_predictions = function(){
    $scope.predictions = $sce.trustAsHtml('No Predictions Yet...')
  }

  $scope.submit = function(){
    console.log('submit')

    if ($scope.num_notes_to_process){
        var req = {
         method: 'POST',
         url: '/predict_static',
         headers: {
           'Content-Type': 'application/json'
         },
         data: JSON.stringify($scope.notes_to_process)
        }

        $http(req).then(function successCallback(response) {
            // this callback will be called asynchronously
            // when the response is available
            console.log(response);
            $scope.predictions = $sce.trustAsHtml(response.data)
          }, function errorCallback(response) {
            // called asynchronously if an error occurs
            // or server returns response with an error status.
          });
    }
    else{
        $scope.predictions = $sce.trustAsHtml('No Notes Selected for Processing')
    }
  }
    
  $scope.json = function(){
    console.log('json')

    if ($scope.num_notes_to_process){
        var req = {
         method: 'POST',
         url: '/predict_json',
         headers: {
           'Content-Type': 'application/json'
         },
         data: JSON.stringify($scope.notes_to_process)
        }

        $http(req).then(function successCallback(response) {
            // this callback will be called asynchronously
            // when the response is available
            console.log(response);
            $scope.json_data = response.data
          }, function errorCallback(response) {
            // called asynchronously if an error occurs
            // or server returns response with an error status.
          });
    }
    else{
        $scope.predictions = $sce.trustAsHtml('No Notes Selected for Processing')
    }
  }

  $scope.clear_note_entry = function(){
    $scope.entry.gender=null;
    $scope.entry.age=null;
    $scope.entry.PO=null;
    $scope.entry.note=null;
    $scope.show_entry_error = false;
  }

  $scope.start_create_note = function(){
    $scope.clear_note_entry()
    $scope.enter_note_active = true;
  }

  $scope.cancel_create_note = function(){
    $scope.enter_note_active = false;
  }

  $scope.accept_create_note = function(){
    console.log('accept code here!');
    console.log($scope.entry);

    if($scope.entry.note==null){
        console.log('issue');
        $scope.show_entry_error = true;

    }
    else{
        console.log('good');
        var copy_of_entry = JSON.parse(JSON.stringify($scope.entry))
        $scope.notes_to_process.push(copy_of_entry);
        $scope.num_notes_to_process = $scope.notes_to_process.length

        // And close the note entry dialogue:
        $scope.clear_note_entry()
        $scope.enter_note_active = false;
    }
  }
})