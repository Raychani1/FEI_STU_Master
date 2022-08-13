<?php
   //A4 - trosku blba funkcia, huh?
   $file = $_GET['file'];
   if(isset($file) && strpos($file, ".php") && file_exists($file))
   {
       include("$file");
   }
   else
   {
       include("index.php");
   }
   ?>
