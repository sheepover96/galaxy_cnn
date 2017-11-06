function unique(arr) {
    var u = {}, a = [];
    for(var i = 0, l = arr.length; i < l; ++i){
        if(!u.hasOwnProperty(arr[i])) {
            a.push(arr[i]);
            u[arr[i]] = 1;
        }
    }
    return a;
}

function output_result(){
    var tf_check_elems = document.getElementsByClassName( "chk" );
    var unknown_check_elems = document.getElementsByClassName( "chk_unknown" );
    var true_output_field = document.getElementById ('result_true');
    var false_output_field = document.getElementById ('result_false');
    var unknown_output_field = document.getElementById ('result_unknown');
    true_output_field.value = ""
    false_output_field.value = ""
    unknown_output_field.value = ""
    var true_list = []
    var false_list = []
    var unknown_list = []
    for (var i = 0; i < unknown_check_elems.length; i++) {
        if(unknown_check_elems[i].checked){
            unknown_list.push(unknown_check_elems[i].dataset.id)
            //unknown_output_field.value += unknown_check_elems[i].dataset.id + '\n'
        }
    }
    for (var i = 0; i < tf_check_elems.length; i++) {
        if(tf_check_elems[i].checked){
            if(!(tf_check_elems[i].dataset.id in unknown_list)){
                true_list.push(tf_check_elems[i].dataset.id)
                //true_output_field.value += tf_check_elems[i].dataset.id + '\n'
            }
        }
        else{
            if(!(tf_check_elems[i].dataset.id in unknown_list)){
                false_list.push(tf_check_elems[i].dataset.id)
                //false_output_field.value += tf_check_elems[i].dataset.id + '\n'
            }
        }
    }
    //var unique_true_list = unique(true_list)
    var unique_true_list = unique(true_list)
    var unique_false_list = unique(false_list)
    var unique_unknown_list = unique(unknown_list)
    for (var i = 0; i < unique_true_list.length; i++) {
        true_output_field.value += unique_true_list[i] + '\n'
    }
    for (var i = 0; i < unique_false_list.length; i++) {
        false_output_field.value += unique_false_list[i] + '\n'
    }
    for (var i = 0; i < unique_unknown_list.length; i++) {
        unknown_output_field.value += unique_unknown_list[i] + '\n'
    }
}