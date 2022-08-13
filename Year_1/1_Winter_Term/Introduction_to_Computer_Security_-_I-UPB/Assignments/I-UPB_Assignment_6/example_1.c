#include <stdio.h>

// Example 1. - The gets() function

// Helping Functions for Example 1.

// Return true if username is not a NULL pointer
int grantAccess(const char* username){
    return username ? 1 : 0;
}

// Print single message
void privilegedAction(){
    printf("Privileged Action in Progress!\n");
}

//// Vulnerable Version

///////////////////////////// Comment Out The Below Region When Running The Safe Version /////////////////////////////

int main () {
    // Character Array for storing username
    char username[10];

    // Allow status
    int allow = 0;

    // Prompt for the user
    printf("Enter your username, please: ");

    // Reference: https://www.tutorialspoint.com/c_standard_library/c_function_gets.htm
    // Vulnerable part of the code. The gets() function, which stands for "Get String" never checks the boundaries
    // ( storing array length ). If you enter a username that fits to this range then everything's gonna be all right,
    // but if you enter a longer username for example : "Hi I'm Johnny Knoxville, welcome to Jackass!" or any String
    // longer than 8 character the program will have undefined behaviour, in this case commonly know as overflow.
    gets(username);

    // If you've entered < 8 characters these statements will be called, otherwise it will be overwritten by the username
    // overflow.
    if (grantAccess(username)) {
        allow = 1;
    }

    if (allow != 0) {
        privilegedAction();
    }

    return 0;
}

///////////////////////////// Comment Out The Above Region When Running The Safe Version /////////////////////////////

//// Safe Version

////////////////////////// Comment Out The Below Region When Running The Vulnerable Version //////////////////////////
//
//#include <stdlib.h>
//#include <string.h>
//#define LENGTH 10
//
//int main () {
//    // Pointers to store username and new line ( '\n' ) memory address
//    char* username, *newlineptr;
//
//    // Allow status
//    int allow = 0;
//
//    // Dynamically Allocate the Memory to store the Username
//    username = malloc(LENGTH * sizeof(*username));
//
//    // If the memory allocation was not successful we should end the program
//    if (!username)
//        return -1;
//
//    // Prompt for the user
//    printf("Enter your username, please: ");
//
//    // Reference: https://www.tutorialspoint.com/c_standard_library/c_function_fgets.htm
//    // The fgets() function reads input form stream. It stops at n-1 ( in our case LENGTH - 1 ) length, or new line
//    // character ( '\n' ), or at the end-of-file if we are reading from a file, whichever comes first.
//    fgets(username,LENGTH, stdin);
//
//    // However the new line character ( '\n' ) is a valid character for the fgets() function we might want to remove
//    // that from the end of our String. We use to the strchr
//    // ( Reference: https://www.tutorialspoint.com/c_standard_library/c_function_strchr.htm ) method, which returns a
//    // character pointer to the first occurrence of the given character or NULL pointer.
//    newlineptr = strchr(username, '\n');
//
//    // This statement will be executed if there was a new line character ( '\n' ) in our username
//    if (newlineptr) {
//        *newlineptr = '\0';
//    }
//
//    // And since this version is not vulnerable to overflows the program execution will continue as originally intended
//    if (grantAccess(username)) {
//        allow = 1;
//    }
//
//    if (allow != 0) {
//        privilegedAction();
//    }
//
//    // Since we've Dynamically Allocated Memory we need to free it before our program ends
//    free(username);
//
//    return 0;
//}

////////////////////////// Comment Out The Above Region When Running The Vulnerable Version //////////////////////////