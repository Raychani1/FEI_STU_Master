package passwordsecurity2;

import java.sql.*;
import java.util.StringTokenizer;

public class Database {

    private static String DB_ADDRESS;
    private static String DB_TABLE;
    private static String DB_USER;
    private static String DB_PASSWD;

    public Database(String address, String table, String user, String password) {
        DB_ADDRESS = address;
        DB_TABLE = table;
        DB_USER = user;
        DB_PASSWD = password;
    }

    final static class MyResult {
        private final boolean first;
        private final String second;

        public MyResult(boolean first, String second) {
            this.first = first;
            this.second = second;
        }

        public boolean getFirst() {
            return first;
        }

        public String getSecond() {
            return second;
        }
    }

    public static boolean checkDatabaseConnection(String url, String user, String pass) {
        Connection connection = null;

        try {
            System.out.println("Establishing Connecting to database...");
            connection = DriverManager.getConnection(url, user, pass);
            System.out.println("Connection Established.");
            return true;
        } catch (SQLException e) {
            System.out.println(e.getMessage());
            return false;
        } finally {
            if (connection != null) {
                try {
                    connection.close();
                } catch (SQLException e) {
                    System.out.println(e.getMessage());
                }
            }
        }
    }

    protected static MyResult add(String text) {
        if (exist(text))
            return new MyResult(false, "Meno uz existuje");

        try (
                Connection connection = DriverManager.getConnection(DB_ADDRESS, DB_USER, DB_PASSWD);
                Statement statement = connection.createStatement()
        ) {
            StringTokenizer st = new StringTokenizer(text, ":");
            String name = st.nextToken();
            String password = st.nextToken();
            String salt = st.nextToken();


            int result = statement.executeUpdate(String.format("INSERT INTO %s (user_name, password, salt) VALUES ('%s', '%s', '%s');", DB_TABLE, name, password, salt));
            if (result == 1)
                return new MyResult(true, "Meno ulozene");
            else
                return new MyResult(false, "Meno sa nepodarilo ulozit");
        } catch (SQLException e) {
            e.printStackTrace();
            return new MyResult(false, e.getMessage());
        }
    }

    protected static MyResult find(String userName) {
        // Check if the user name is not in the Database
        if (!exist(userName))
            return new MyResult(false, "Meno sa nenaslo");

        // Try to establish a Connection
        try (
                Connection connection = DriverManager.getConnection(DB_ADDRESS, DB_USER, DB_PASSWD);
                Statement statement = connection.createStatement()
        ) {
            // Get the User's data

            ResultSet rs = statement.executeQuery(String.format("SELECT user_name, password, salt FROM %s WHERE user_name = '%s';", DB_TABLE, userName));

            // Return if something was found
            if (rs.next()) {
                String name = rs.getString("user_name");
                String password = rs.getString("password");
                String salt = rs.getString("salt");

                return new MyResult(true, name + ":" + password + ":" + salt);
            }
            return new MyResult(false, "Meno sa nenaslo");

        } catch (SQLException e) {
            e.printStackTrace();
            return new MyResult(false, e.getMessage());
        }
    }

    protected static boolean exist(String userName) {
        // Try to establish a Connection
        try (
                Connection connection = DriverManager.getConnection(DB_ADDRESS, DB_USER, DB_PASSWD);
                Statement statement = connection.createStatement()
        ) {
            // Check for user name in Database
            ResultSet rs = statement.executeQuery(String.format("SELECT count(user_name) AS names FROM %s WHERE user_name = '%s';", DB_TABLE, userName));

            // Return False if there is no user with this name
            if (rs.next()) {
                return rs.getInt("names") != 0;
            } else return true;
        } catch (SQLException e) {
            e.printStackTrace();
            return true;
        }
    }

}
