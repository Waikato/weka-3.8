/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ExtensionFileFilterWithClass.java
 * Copyright (C) 2020 University of Waikato, Hamilton, NZ
 */

package weka.gui;

/**
 * File filter that stores an associated class alongside name and extension(s).
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class ExtensionFileFilterWithClass
  extends ExtensionFileFilter {

  /** the underlying class. */
  protected Class m_FilterClass;

  /**
   * Creates the ExtensionFileFilterWithClass
   *
   * @param extension   the extension of accepted files.
   * @param description a text description of accepted files.
   * @param filterClass the underlying class
   */
  public ExtensionFileFilterWithClass(String extension, String description, Class filterClass) {
    super(extension, description);
    if (filterClass == null)
      throw new IllegalArgumentException("Filter class cannot be null!");
    m_FilterClass = filterClass;
  }

  /**
   * Creates an ExtensionFileFilterWithClass that accepts files that have any of the
   * extensions contained in the supplied array.
   *
   * @param extensions  an array of acceptable file extensions (as Strings).
   * @param description a text description of accepted files.
   * @param filterClass the underlying class
   */
  public ExtensionFileFilterWithClass(String[] extensions, String description, Class filterClass) {
    super(extensions, description);
    if (filterClass == null)
      throw new IllegalArgumentException("Filter class cannot be null!");
    m_FilterClass = filterClass;
  }

  /**
   * Returns the underlying class.
   *
   * @return		the class
   */
  public Class getFilterClass() {
    return m_FilterClass;
  }

  /**
   * Creates a new instance of the underlying class.
   *
   * @return		the object
   */
  public Object newInstance() {
    try {
      return m_FilterClass.newInstance();
    }
    catch (Exception e) {
      System.err.println("Failed to instantiate filter class: " + m_FilterClass.getName());
      e.printStackTrace();
      return null;
    }
  }
}
