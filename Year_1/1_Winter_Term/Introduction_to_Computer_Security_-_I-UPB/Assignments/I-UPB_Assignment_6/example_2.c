#include <stdio.h>
#include <string.h>

// Example 2. -

// Helping Functions for Example 2.

int target;

//// Vulnerable Version

///////////////////////////// Comment Out The Below Region When Running The Safe Version /////////////////////////////

//void vulnerability(char *string){
//    printf(string);
//
//    if(target){
//        printf("Hackers in the system!");
//    }
//}

///////////////////////////// Comment Out The Above Region When Running The Safe Version /////////////////////////////

//// Safe Version

////////////////////////// Comment Out The Below Region When Running The Vulnerable Version //////////////////////////

void vulnerability(char *string){
    printf("%s", string);

    if(target){
        printf("Hackers in the system!");
    }
}

////////////////////////// Comment Out The Above Region When Running The Vulnerable Version //////////////////////////

int main(int argc, char **argv) {

    vulnerability(argv[1]);

    return 0;
}
