<?php
function verify_login(){
    global $db;

    // Create Prepared Statement
    $sql = $db->prepare("SELECT id,name,password FROM admins WHERE name=? AND password=? LIMIT 1");
    
    // Get Username and Password Hash Values
    $username = $_POST['name'];
    $passwd = hash("sha512", $_POST['pass']);

    // Bind values to Prepared Statement
    $sql->bind_param("ss", $username, $passwd);

    // Execute Statement
    $sql->execute();
    
    // Get Results
    $data = $sql->get_result()->fetch_row();
     
    if(!empty($data)){
	// Set Session Values
        $_SESSION['id']  = $data[0];
        $_SESSION['name'] = $data[1];
        $_SESSION['session_id'] = session_id();
        return true;
    }else{
        return false;
    }
}

//echo hash("sha512","student");

if(@$_POST['logIN']){
    if(verify_login()) {
        header('LOCATION: index.php');
    }else{
        $error = "Wrong name or password!! Pls try it again!!";
    }
}

// SOURCE:
// https://www.wikihow.com/Prevent-Cross-Site-Request-Forgery-(CSRF)-Attacks-in-PHP

session_start();
include 'csrf.php';

$csrf = new csrf();


// Generate Token Id and Valid
$token_id = $csrf->get_token_id();
$token_value = $csrf->get_token($token_id);

// Generate Random Form Names
$form_names = $csrf->form_names(array('name', 'pass'), false);


if(isset($_POST["name"], $_POST["pass"])) {
	// Check if token id and token value are valid.
	if($csrf->check_valid('post')) {
		// Get the Form Variables.
		$user = $_POST["name"];
		$password = $_POST["pass"];
		
		// Form Function Goes Here
	}
	// Regenerate a new random value for the form.
	$form_names = $csrf->form_names(array("name", "pass"), true);
}

if(!isLogin()){?>
<div style="width:20%;">
    <?=@$error?>
    <form method="post" name="login">
        <label>Meno</label>
        <input name="name" value="" type="text" placeholder="LamaCoder" autofocus />
        <label>Heslo</label>
        <input name="pass" value="" type="password" placeholder="********" />
        <br />
        <button class="button" name="logIN" value="1">Prihlasiť</button>
    </form>
</div>
<?}else{?>
    <div style="width:20%;">
        <?=@$error?>
        <a href="./?page=logout.php"><button class="button">Odhlásiť sa</button></a>
    </div>
<?}?>
