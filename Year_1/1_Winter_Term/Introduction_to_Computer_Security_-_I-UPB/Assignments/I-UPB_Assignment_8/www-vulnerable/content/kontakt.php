<?php

// SOURCE:
// https://www.wikihow.com/Prevent-Cross-Site-Request-Forgery-(CSRF)-Attacks-in-PHP

session_start();
include 'csrf.php';

$csrf = new csrf();

// Generate Token Id and Valid
$token_id = $csrf->get_token_id();
$token_value = $csrf->get_token($token_id);

// Generate Random Form Names
$form_names = $csrf->form_names(array('name', 'email', 'message'), false);


if(isset($_POST['name'], $_POST['email'], $_POST['message'])) {
	// Check if token id and token value are valid.
	if($csrf->check_valid('post')) {
		// Get the Form Variables.
		$user = $_POST['name'];
		$email = $_POST['email'];
		$message = $_POST['message'];
		
		// Form Function Goes Here
	}
	// Regenerate a new random value for the form.
	$form_names = $csrf->form_names(array('name', 'email', 'message'), true);
}

$action=$_REQUEST['action'];
if ($action=="")    /* display the contact form */
    {
    ?>
    <form  action="" method="POST" enctype="multipart/form-data">
    <input type="hidden" name="action" value="submit">
    Your name:<br>
    <input name="name" type="text" value="" size="30"/><br>
    Your email:<br>
    <input name="email" type="text" value="" size="30"/><br>
    Your message:<br>
    <textarea name="message" rows="7" cols="30"></textarea><br>
    <input type="submit" value="Send email"/>
    </form>
    <?php
    } 
else                /* send the submitted data */
    {
    $name=$_REQUEST['name'];
    $email=$_REQUEST['email'];
    $message=$_REQUEST['message'];
    if (($name=="")||($email=="")||($message==""))
        {
        echo "All fields are required, please fill <a href=\"\">the form</a> again.";
        }
    else{        
        $from="From: $name<$email>\r\nReturn-path: $email";
        $subject="Message sent using your contact form";
        mail("youremail@yoursite.com", $subject, $message, $from);
        echo "Email sent!";
        }
    }  
?> 
