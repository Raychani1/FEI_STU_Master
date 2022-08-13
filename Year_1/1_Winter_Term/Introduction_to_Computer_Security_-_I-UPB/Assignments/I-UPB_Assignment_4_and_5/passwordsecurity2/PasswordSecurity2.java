package passwordsecurity2;

import javax.swing.*;

public class PasswordSecurity2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.out.println("Please specify the Database Address ( URL ), Table,  User and Password!");
        } else {
            Database database = new Database(args[0], args[1], args[2], args[3]);

            if (Database.checkDatabaseConnection(args[0], args[2], args[3])) {
                GUI okno = new GUI();
                okno.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                okno.setVisible(true);
                okno.setResizable(false);
                okno.setLocationRelativeTo(null);
            } else {
                System.out.println("Cannot connect to database.");
            }
        }
    }
}
